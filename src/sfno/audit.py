import json
from pathlib import Path

from sfno.checkpoints import load_models
from sfno.experiments import ABLATIONS
from sfno.model import count_parameters
from sfno.runtime import sha256


def inspect_checkpoints(config, device, directory, mapping=None):
    models, sources = load_models(config, device, directory, mapping)
    return {
        ablation.key: {
            **sources[ablation.key],
            **count_parameters(models[ablation.key]),
            "state_dict_keys": list(models[ablation.key].state_dict()),
        }
        for ablation in ABLATIONS
    }


def verify_artifacts(directory):
    directory = Path(directory)
    manifests = sorted(directory.rglob("manifest.json"))
    if not manifests:
        raise FileNotFoundError(f"No manifests found under {directory}.")
    results = []
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        base = manifest_path.parent
        for artifact in manifest["artifacts"]:
            path = (base / artifact["path"]).resolve()
            if not path.is_relative_to(base.resolve()):
                raise ValueError(f"Artifact path escapes its run directory: {artifact['path']}")
            valid = path.is_file() and sha256(path) == artifact["sha256"]
            results.append(
                {"manifest": str(manifest_path), "artifact": artifact["path"], "valid": valid}
            )
    return {
        "valid": all(row["valid"] for row in results),
        "checked": len(results),
        "files": results,
    }


def compare_reported(evaluation_path, reference_path):
    evaluation_path = Path(evaluation_path)
    path = evaluation_path / "metrics.json" if evaluation_path.is_dir() else evaluation_path
    actual = json.loads(path.read_text(encoding="utf-8"))
    expected = json.loads(Path(reference_path).read_text(encoding="utf-8"))
    if expected.get("kind") != "reported_metrics":
        raise ValueError("Expected the recorded study-metrics reference file.")
    rows = []
    for ablation in ABLATIONS:
        for metric, value in expected["metrics"][ablation.key].items():
            precision = 4 if metric == "gaussian_crps" else 2
            measured = actual[ablation.key][metric]
            rows.append(
                {
                    "model": ablation.key,
                    "metric": metric,
                    "recorded": value,
                    "computed": measured,
                    "difference": measured - value,
                    "matches_displayed_precision": f"{measured:.{precision}f}"
                    == f"{value:.{precision}f}",
                }
            )
    return {
        "reference_kind": expected["kind"],
        "comparisons": rows,
        "all_match_displayed_precision": all(row["matches_displayed_precision"] for row in rows),
    }
