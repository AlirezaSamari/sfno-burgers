import json
from dataclasses import replace

import numpy as np
import pytest
import reference_implementation as reference
import torch
from torch.utils.data import DataLoader, TensorDataset

from sfno.audit import verify_artifacts
from sfno.checkpoints import checkpoint_paths, load_evaluated_model
from sfno.config import Config
from sfno.data import generate_datasets, load_dataset
from sfno.evaluate import evaluate_models, load_evaluation
from sfno.experiments import ABLATIONS
from sfno.forcing import compute_xi_grid
from sfno.inverse import recover_coefficients
from sfno.model import make_model
from sfno.runtime import capture_rng, load_torch, restore_rng, save_torch, seed_everything
from sfno.train import train_models


def exact(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_rng_roundtrip_is_weights_only_loadable(tmp_path):
    seed_everything(6)
    state = capture_rng()
    path = tmp_path / "rng.pt"
    save_torch(path, state)
    expected = torch.randn(7), np.random.randn(7)
    restore_rng(load_torch(path), torch.device("cpu"), "cpu")
    exact(torch.randn(7), expected[0])
    np.testing.assert_array_equal(np.random.randn(7), expected[1])


def test_unknown_configuration_and_mode_limits(smoke):
    values = smoke.to_dict()
    values["training"]["learning_rte"] = 0.1
    with pytest.raises(TypeError):
        Config.from_dict(values)
    with pytest.raises(ValueError, match="Fourier"):
        replace(smoke, architecture=replace(smoke.architecture, modes=90)).validate()


def test_checkpoint_alias_collision_requires_mapping(tmp_path):
    for ablation in ABLATIONS:
        save_torch(tmp_path / ablation.filename, {"weight": torch.ones(1)})
    canonical = tmp_path / "model_hybrid_s-pino.pt"
    alias = tmp_path / "model_hybrid_pino.pt"
    alias.write_bytes(canonical.read_bytes())
    assert checkpoint_paths(tmp_path)["residual"] == canonical
    save_torch(alias, {"weight": torch.zeros(1)})
    with pytest.raises(ValueError, match="Conflicting"):
        checkpoint_paths(tmp_path)
    mapping = tmp_path / "mapping.json"
    mapping.write_text(json.dumps({"residual": alias.name}))
    assert checkpoint_paths(tmp_path, mapping)["residual"] == alias


def test_training_resume_reproduces_uninterrupted_run(tmp_path, smoke):
    device = torch.device("cpu")
    data = generate_datasets(smoke, device, tmp_path / "data")
    train_models(data, tmp_path / "continuous", device)
    train_models(data, tmp_path / "resumed", device, stop_after_epochs=1)
    train_models(data, tmp_path / "resumed", device, resume=True)
    for ablation in ABLATIONS:
        first = load_torch(tmp_path / "continuous/checkpoints" / ablation.filename)
        second = load_torch(tmp_path / "resumed/checkpoints" / ablation.filename)
        for key in first:
            exact(first[key], second[key])
    continuous = json.loads((tmp_path / "continuous/histories.json").read_text())
    resumed = json.loads((tmp_path / "resumed/histories.json").read_text())
    assert continuous == resumed
    evaluate_models(data, tmp_path / "resumed/checkpoints", tmp_path / "evaluation", device)
    bundle, config = load_evaluation(tmp_path / "evaluation")
    assert set(bundle["metrics"]) == {ablation.key for ablation in ABLATIONS}
    assert config.fingerprint() == smoke.fingerprint()
    assert verify_artifacts(tmp_path)["valid"]
    wrong = tmp_path / "wrong.pt"
    save_torch(wrong, {"weight": torch.zeros(1)})
    with pytest.raises(ValueError, match="differs"):
        load_evaluated_model(bundle, config, device, wrong)
    with pytest.raises(FileExistsError):
        generate_datasets(smoke, device, tmp_path / "data")
    (tmp_path / "evaluation/table1.csv").write_text("modified\n")
    assert not verify_artifacts(tmp_path)["valid"]


def test_five_model_training_order_matches_reference(tmp_path, smoke):
    device = torch.device("cpu")
    config = replace(
        smoke,
        architecture=replace(smoke.architecture, projection_width=128),
        training=replace(smoke.training, epochs=1, scheduler_step=100),
    )
    path = generate_datasets(config, device, tmp_path / "data")
    payload, _ = load_dataset(path)
    train_models(path, tmp_path / "training", device)
    restore_rng(payload["rng_after_data"], device, payload["device"])
    models, optimizers, schedulers = [], [], []
    for _ in ABLATIONS:
        model = reference.StochasticFNO(modes=3, width=4, num_layers=2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
        models.append(model)
        optimizers.append(optimizer)
        schedulers.append(scheduler)
    data = payload["train"]
    loader = DataLoader(
        TensorDataset(data["u0"], data["xi"], data["target"]), batch_size=2, shuffle=True
    )
    expected_histories = {}
    for ablation, model, optimizer, scheduler in zip(ABLATIONS, models, optimizers, schedulers):
        total = 0.0
        for initial, forcing, target in loader:
            optimizer.zero_grad()
            predicted = model(data["x"], data["t"], initial, forcing)
            if ablation.residual:
                loss_data = torch.mean((predicted - target) ** 2)
                loss_phys = reference.compute_pinn_pde_loss(
                    predicted, forcing, initial, data["x"], data["t"], reference.nu, reference.sigma
                )
                loss = loss_data + 0.1 * loss_phys
            else:
                loss = reference.compute_topological_spectral_loss(
                    predicted,
                    target,
                    0.1 if ablation.spectral else 0.0,
                    0.05 if ablation.increment else 0.0,
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
        scheduler.step()
        expected_histories[ablation.key] = [total / len(loader)]
        actual = load_torch(tmp_path / "training/checkpoints" / ablation.filename)
        for key, value in model.state_dict().items():
            exact(actual[key], value)
    actual_histories = json.loads((tmp_path / "training/histories.json").read_text())
    assert actual_histories == expected_histories


def test_dataset_config_mismatch_is_rejected(tmp_path, smoke):
    path = generate_datasets(smoke, torch.device("cpu"), tmp_path / "data")
    with pytest.raises(ValueError, match="configuration"):
        load_dataset(path, replace(smoke, seed=43))


def test_inverse_reports_consistent_pre_and_post_update_iterates(smoke):
    device = torch.device("cpu")
    config = replace(smoke, inverse=replace(smoke.inverse, iterations=3, scheduler_step=200))
    torch.manual_seed(29)
    model = make_model(config, device)
    x, t = torch.linspace(0, 1, 8), torch.linspace(0, 1, 8)
    initial, observed = torch.randn(1, 8), torch.randn(1, 8, 8)
    result = recover_coefficients(model, initial, observed, x, t, config)
    torch.manual_seed(42)
    z = torch.randn(1, 9, requires_grad=True)
    optimizer = torch.optim.Adam([z], lr=0.05)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
    for _ in range(3):
        optimizer.zero_grad()
        xi = compute_xi_grid(z, x, t, config.physics)
        prediction = model(x, t, initial, xi)
        loss_data = torch.mean((prediction - observed) ** 2)
        loss = loss_data + 5e-4 * torch.mean(z**2)
        loss.backward()
        optimizer.step()
        scheduler.step()
    exact(result["last_evaluated"]["xi"], xi)
    exact(result["last_evaluated"]["predicted"], prediction)
    exact(result["after_final_update"]["coefficients"], z)
    assert result["last_evaluated"]["data_loss"] == loss_data.item()
    with torch.no_grad():
        for key in ("last_evaluated", "after_final_update"):
            value = result[key]
            forcing = compute_xi_grid(value["coefficients"], x, t, config.physics)
            exact(value["xi"], forcing)
            exact(value["predicted"], model(x, t, initial, forcing))
    assert not torch.equal(
        result["last_evaluated"]["coefficients"], result["after_final_update"]["coefficients"]
    )
