import logging

import torch
from torch.utils.data import DataLoader, TensorDataset

from sfno.data import dataset_source, load_dataset
from sfno.experiments import ABLATIONS, objective_weights
from sfno.losses import composite_loss, pde_residual_loss
from sfno.model import make_model
from sfno.runtime import (
    capture_rng,
    load_torch,
    prepare_output,
    restore_rng,
    save_torch,
    to_cpu,
    write_json,
    write_manifest,
)

LOGGER = logging.getLogger(__name__)


def train_models(data_path, output, device, resume=False, stop_after_epochs=None):
    payload, config = load_dataset(data_path)
    output = prepare_output(output, resume=resume)
    latest = output / "training_state.pt"
    if resume and not latest.is_file():
        raise FileNotFoundError(f"No training state to resume: {latest}")
    source = dataset_source(data_path)
    restore_rng(payload["rng_after_data"], device, payload["device"])
    training = config.training
    models, optimizers, schedulers, histories = {}, {}, {}, {}
    for ablation in ABLATIONS:
        model = make_model(config, device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=training.learning_rate,
            weight_decay=training.weight_decay,
            betas=(training.adam_beta1, training.adam_beta2),
            eps=training.adam_epsilon,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=training.scheduler_step,
            gamma=training.scheduler_gamma,
        )
        models[ablation.key], optimizers[ablation.key] = model, optimizer
        schedulers[ablation.key], histories[ablation.key] = scheduler, []
    active_model, next_epoch = 0, 0
    if resume:
        state = load_torch(latest)
        if state.get("kind") != "sfno_training_state":
            raise ValueError("Invalid training state.")
        if (
            state["config_sha256"] != config.fingerprint()
            or state["dataset_sha256"] != source["sha256"]
        ):
            raise ValueError("The training state does not match the configuration and dataset.")
        for ablation in ABLATIONS:
            key = ablation.key
            models[key].load_state_dict(state["models"][key], strict=True)
            optimizers[key].load_state_dict(state["optimizers"][key])
            schedulers[key].load_state_dict(state["schedulers"][key])
        histories = state["histories"]
        active_model, next_epoch = state["active_model"], state["next_epoch"]
        restore_rng(state["rng"], device, state["device"])
    train = payload["train"]
    loader = DataLoader(
        TensorDataset(train["u0"], train["xi"], train["target"]),
        batch_size=training.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
        pin_memory=False,
    )
    x_grid, t_grid = train["x"].to(device), train["t"].to(device)
    write_json(output / "config.json", config.to_dict())
    completed_this_call = 0

    def save_progress(model_index, epoch_index):
        state = {
            "kind": "sfno_training_state",
            "schema_version": 1,
            "config_sha256": config.fingerprint(),
            "dataset_sha256": source["sha256"],
            "device": str(device),
            "active_model": model_index,
            "next_epoch": epoch_index,
            "models": {key: to_cpu(model.state_dict()) for key, model in models.items()},
            "optimizers": {key: to_cpu(opt.state_dict()) for key, opt in optimizers.items()},
            "schedulers": {key: scheduler.state_dict() for key, scheduler in schedulers.items()},
            "histories": histories,
            "rng": capture_rng(),
        }
        save_torch(latest, state)
        write_json(output / "histories.json", histories)

    for model_index in range(active_model, len(ABLATIONS)):
        ablation = ABLATIONS[model_index]
        key = ablation.key
        model, optimizer, scheduler = models[key], optimizers[key], schedulers[key]
        weights = objective_weights(ablation, training)
        start_epoch = next_epoch if model_index == active_model else 0
        LOGGER.info("Training %s", ablation.name)
        for epoch in range(start_epoch, training.epochs):
            model.train()
            epoch_loss = 0.0
            for initial, forcing, target in loader:
                initial, forcing, target = initial.to(device), forcing.to(device), target.to(device)
                optimizer.zero_grad()
                predicted = model(x_grid, t_grid, initial, forcing)
                if ablation.residual:
                    data_loss = torch.mean((predicted - target) ** 2)
                    physical_loss = pde_residual_loss(
                        predicted,
                        forcing,
                        x_grid,
                        t_grid,
                        config.physics.viscosity,
                        config.physics.noise_scale,
                    )
                    loss = data_loss + weights[2] * physical_loss
                else:
                    loss = composite_loss(
                        predicted, target, weights[0], weights[1], training.spectral_floor
                    )
                if not torch.isfinite(loss).item():
                    raise FloatingPointError(f"Nonfinite loss for {key} at epoch {epoch}.")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), training.gradient_clip)
                optimizer.step()
                epoch_loss += loss.item()
            scheduler.step()
            histories[key].append(epoch_loss / len(loader))
            completed_this_call += 1
            if epoch % 100 == 0 or epoch == training.epochs - 1:
                LOGGER.info("%s epoch %04d loss %.6e", key, epoch, histories[key][-1])
            finished = epoch + 1 == training.epochs
            if finished:
                save_torch(output / "checkpoints" / ablation.filename, to_cpu(model.state_dict()))
            should_stop = stop_after_epochs is not None and completed_this_call >= stop_after_epochs
            if finished or should_stop or (epoch + 1) % training.checkpoint_interval == 0:
                save_progress(
                    model_index + 1 if finished else model_index, 0 if finished else epoch + 1
                )
            if should_stop:
                LOGGER.info("Stopped at an epoch boundary; continue with --resume.")
                return output
    artifacts = [output / "checkpoints" / ablation.filename for ablation in ABLATIONS]
    artifacts += [output / "histories.json", output / "config.json"]
    write_manifest(
        output,
        "train",
        config,
        device,
        [source],
        artifacts,
        {"completed": True, "epochs_per_model": training.epochs},
    )
    return output
