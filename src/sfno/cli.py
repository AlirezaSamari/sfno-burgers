import argparse
import json
import logging
import sys
from pathlib import Path

import torch

from sfno.audit import compare_reported, inspect_checkpoints, verify_artifacts
from sfno.benchmark import benchmark_experiment
from sfno.config import load_config
from sfno.data import generate_datasets
from sfno.evaluate import evaluate_models
from sfno.inverse import inverse_experiment
from sfno.model import count_parameters, make_model
from sfno.resolution import resolution_experiment
from sfno.runtime import prepare_output, resolve_device
from sfno.train import train_models


def positive_integer(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Expected a positive integer.")
    return number


def add_runtime(parser, default="auto"):
    parser.add_argument("--device", default=default, help="auto, cpu or cuda:N")
    parser.add_argument(
        "--threads", type=positive_integer, help="Override the PyTorch CPU thread count"
    )


def add_config(parser):
    parser.add_argument("--config", type=Path, help="Custom JSON configuration")
    parser.add_argument("--profile", choices=("paper", "smoke"), default="paper")


def build_parser():
    parser = argparse.ArgumentParser(
        prog="sfno", description="Stochastic Burgers Fourier-operator experiments"
    )
    parser.add_argument("--verbose", action="store_true")
    commands = parser.add_subparsers(dest="command", required=True)
    settings = commands.add_parser("settings", help="Print every experiment setting")
    add_config(settings)
    inspect = commands.add_parser("inspect", help="Inspect architecture and parameter counts")
    add_config(inspect)
    add_runtime(inspect, "cpu")
    generate = commands.add_parser(
        "generate", help="Generate training then test data and save the random state"
    )
    add_config(generate)
    add_runtime(generate)
    generate.add_argument("--output", type=Path, required=True)
    train = commands.add_parser("train", help="Train the five objectives in the configured order")
    add_runtime(train)
    train.add_argument("--data", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--resume", action="store_true")
    train.add_argument(
        "--stop-after", type=positive_integer, help="Stop after this many additional model epochs"
    )
    evaluate = commands.add_parser(
        "evaluate", help="Evaluate all five checkpoints and write both metric tables"
    )
    add_runtime(evaluate)
    evaluate.add_argument("--data", type=Path, required=True)
    evaluate.add_argument("--checkpoints", type=Path, required=True)
    evaluate.add_argument("--mapping", type=Path)
    evaluate.add_argument("--output", type=Path, required=True)
    for name in ("resolution", "inverse", "benchmark"):
        experiment = commands.add_parser(
            name, help=f"Run the {name} experiment using an evaluated S-FNO"
        )
        add_runtime(experiment)
        experiment.add_argument("--evaluation", type=Path, required=True)
        experiment.add_argument(
            "--checkpoint", type=Path, help="Relocated checkpoint with the same recorded SHA-256"
        )
        experiment.add_argument("--output", type=Path, required=True)
        if name == "resolution":
            experiment.add_argument("--include-unpaired-diagnostic", action="store_true")
    plot = commands.add_parser("plot", help="Render editable PDF line plots and PNG contours")
    plot.add_argument("--evaluation", type=Path, required=True)
    plot.add_argument("--resolution", type=Path)
    plot.add_argument("--inverse", type=Path)
    plot.add_argument("--training", type=Path)
    plot.add_argument("--output", type=Path, required=True)
    plot.add_argument("--no-legends", action="store_true")
    for name in ("reproduce", "smoke"):
        pipeline = commands.add_parser(
            name,
            help="Run the complete workflow"
            if name == "reproduce"
            else "Run the reduced integration check",
        )
        add_runtime(pipeline, "cpu" if name == "smoke" else "auto")
        pipeline.add_argument("--output", type=Path, required=True)
        if name == "reproduce":
            add_config(pipeline)
    checkpoints = commands.add_parser(
        "checkpoints", help="Check names, hashes and state-dictionary compatibility"
    )
    add_config(checkpoints)
    add_runtime(checkpoints, "cpu")
    checkpoints.add_argument("--directory", type=Path, required=True)
    checkpoints.add_argument("--mapping", type=Path)
    verify = commands.add_parser("verify", help="Verify the hashes in run manifests")
    verify.add_argument("--run", type=Path, required=True)
    compare = commands.add_parser(
        "compare", help="Compare computed metrics with recorded study results"
    )
    compare.add_argument("--evaluation", type=Path, required=True)
    compare.add_argument("--reference", type=Path, required=True)
    return parser


def run_pipeline(config, output, device):
    from sfno.plotting import render_figures

    output = prepare_output(output)
    data = generate_datasets(config, device, output / "data")
    train_models(data, output / "training", device)
    evaluate_models(data, output / "training/checkpoints", output / "evaluation", device)
    resolution_experiment(output / "evaluation", output / "resolution", device)
    inverse_experiment(output / "evaluation", output / "inverse", device)
    render_figures(
        output / "evaluation",
        output / "figures",
        output / "resolution",
        output / "inverse",
        output / "training",
    )
    return verify_artifacts(output)


def main(argv=None):
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s"
    )
    if getattr(args, "threads", None):
        torch.set_num_threads(args.threads)
    elif args.command == "smoke":
        torch.set_num_threads(1)
    try:
        result = dispatch(args)
    except (
        ValueError,
        TypeError,
        KeyError,
        FileNotFoundError,
        FileExistsError,
        FloatingPointError,
        RuntimeError,
    ) as error:
        logging.error("%s", error)
        if args.verbose:
            raise
        return 1
    if result is not None:
        print(json.dumps(result, indent=2, allow_nan=False, default=str))
    if args.command == "verify" and not result["valid"]:
        return 1
    return 0


def dispatch(args):
    command = args.command
    if command == "settings":
        return load_config(args.config, args.profile).to_dict()
    if command == "verify":
        return verify_artifacts(args.run)
    if command == "compare":
        return compare_reported(args.evaluation, args.reference)
    if command == "plot":
        from sfno.plotting import render_figures

        return {
            "figures": render_figures(
                args.evaluation,
                args.output,
                args.resolution,
                args.inverse,
                args.training,
                args.no_legends,
            )
        }
    device = resolve_device(args.device)
    if command == "inspect":
        config = load_config(args.config, args.profile)
        return {"config": config.to_dict(), **count_parameters(make_model(config, device))}
    if command == "checkpoints":
        return inspect_checkpoints(
            load_config(args.config, args.profile), device, args.directory, args.mapping
        )
    if command == "generate":
        return {
            "dataset": generate_datasets(
                load_config(args.config, args.profile), device, args.output
            )
        }
    if command == "train":
        return {
            "training": train_models(args.data, args.output, device, args.resume, args.stop_after)
        }
    if command == "evaluate":
        return {
            "evaluation": evaluate_models(
                args.data, args.checkpoints, args.output, device, args.mapping
            )
        }
    if command == "resolution":
        return {
            "resolution": resolution_experiment(
                args.evaluation,
                args.output,
                device,
                args.checkpoint,
                args.include_unpaired_diagnostic,
            )
        }
    if command == "inverse":
        return {
            "inverse": inverse_experiment(args.evaluation, args.output, device, args.checkpoint)
        }
    if command == "benchmark":
        return benchmark_experiment(args.evaluation, args.output, device, args.checkpoint)
    if command in {"reproduce", "smoke"}:
        config = (
            load_config(profile="smoke")
            if command == "smoke"
            else load_config(args.config, args.profile)
        )
        return run_pipeline(config, args.output, device)
    raise ValueError(f"Unknown command: {command}")


if __name__ == "__main__":
    sys.exit(main())
