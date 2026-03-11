from __future__ import annotations

# python .\run_optuna_search.py --study-config configs\optuna_loso_ds002336.yaml --mode finetune_only
# python .\run_optuna_search.py --study-config configs\optuna_loso_ds002336.yaml --mode contrastive_only

"""基于 Optuna 的自动化超参搜索与结果汇总入口。"""

import argparse
import csv
import copy
import json
import math
import os
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Optuna automation for EEG-fMRI-Contrastive")
    parser.add_argument("--study-config", type=str, required=True, help="YAML file that defines command, search space, and metric source.")
    parser.add_argument("--mode", type=str, default="", help="Optional study mode, such as full, finetune_only, or contrastive_only.")
    parser.add_argument("--n-trials", type=int, default=None, help="Override study.n_trials from YAML.")
    parser.add_argument("--timeout", type=int, default=None, help="Override study.timeout in seconds.")
    parser.add_argument("--study-name", type=str, default="", help="Override study.name from YAML.")
    parser.add_argument("--output-dir", type=str, default="", help="Override study.output_dir from YAML.")
    parser.add_argument("--summary-only", action="store_true", help="Only regenerate summaries from an existing study database.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop immediately when a trial command fails.")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Study config must be a mapping: {path}")
    return payload


def assign_nested_value(payload: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cursor = payload
    for key in parts[:-1]:
        next_value = cursor.get(key)
        if next_value is None:
            next_value = {}
            cursor[key] = next_value
        if not isinstance(next_value, dict):
            raise ValueError(f"Cannot override nested key '{dotted_key}' because '{key}' is not a mapping")
        cursor = next_value
    cursor[parts[-1]] = value


def resolve_path(path_value: str, *, base_dir: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else (base_dir / path).resolve()


def normalize_study_config(raw: dict[str, Any], args: argparse.Namespace, config_path: Path) -> dict[str, Any]:
    study_cfg = dict(raw.get("study", {}))
    metric_cfg = dict(raw.get("metric", {}))
    parameters_cfg = dict(raw.get("parameters", {}))
    modes_cfg = dict(raw.get("modes", {}))
    if not study_cfg:
        raise ValueError("Missing 'study' section in study config")
    if not metric_cfg:
        raise ValueError("Missing 'metric' section in study config")
    if not parameters_cfg:
        raise ValueError("Missing 'parameters' section in study config")

    config_dir = config_path.resolve().parent
    study_name = args.study_name.strip() or str(study_cfg.get("name", config_path.stem)).strip()
    if not study_name:
        raise ValueError("study.name must not be empty")

    output_dir_value = args.output_dir.strip() or str(study_cfg.get("output_dir", f"outputs/optuna/{study_name}"))
    output_dir = resolve_path(output_dir_value, base_dir=PROJECT_ROOT)
    command = study_cfg.get("command", [])
    if not isinstance(command, list) or not command:
        raise ValueError("study.command must be a non-empty string list")

    static_args = study_cfg.get("static_args", [])
    if not isinstance(static_args, list):
        raise ValueError("study.static_args must be a string list")

    cwd_value = str(study_cfg.get("cwd", "."))
    cwd = resolve_path(cwd_value, base_dir=PROJECT_ROOT)
    storage_value = str(study_cfg.get("storage", "")).strip()
    if storage_value:
        storage = storage_value
        storage_path = None
    else:
        storage_path = output_dir / "study.db"
        storage = f"sqlite:///{storage_path.as_posix()}"

    metric_path = metric_cfg.get("path")
    if not metric_path:
        raise ValueError("metric.path must be configured")

    normalized = {
        "config_path": config_path.resolve(),
        "config_dir": config_dir,
        "study_name": study_name,
        "mode": "",
        "direction": str(study_cfg.get("direction", "maximize")).strip().lower(),
        "n_trials": int(args.n_trials if args.n_trials is not None else study_cfg.get("n_trials", 20)),
        "timeout": args.timeout if args.timeout is not None else study_cfg.get("timeout", None),
        "output_dir": output_dir,
        "storage": storage,
        "storage_path": storage_path,
        "command": [str(item) for item in command],
        "static_args": [str(item) for item in static_args],
        "cwd": cwd,
        "output_arg": str(study_cfg.get("output_arg", "")).strip(),
        "sampler": dict(study_cfg.get("sampler", {})),
        "metric": {
            "type": str(metric_cfg.get("type", "json")).strip().lower(),
            "path": str(metric_path),
            "key": str(metric_cfg.get("key", "")).strip(),
            "column": str(metric_cfg.get("column", "")).strip(),
            "row_filter": dict(metric_cfg.get("row_filter", {})),
        },
        "parameters": parameters_cfg,
        "runtime_configs": {},
    }

    runtime_cfg = dict(raw.get("runtime_configs", {}))
    if runtime_cfg:
        normalized["runtime_configs"] = {
            "train_base": resolve_path(str(runtime_cfg.get("train_base", "")), base_dir=PROJECT_ROOT) if str(runtime_cfg.get("train_base", "")).strip() else None,
            "finetune_base": resolve_path(str(runtime_cfg.get("finetune_base", "")), base_dir=PROJECT_ROOT) if str(runtime_cfg.get("finetune_base", "")).strip() else None,
            "train_arg": str(runtime_cfg.get("train_arg", "-TrainConfig")).strip(),
            "finetune_arg": str(runtime_cfg.get("finetune_arg", "-FinetuneConfig")).strip(),
        }

    selected_mode = args.mode.strip() or str(study_cfg.get("default_mode", "")).strip()
    if selected_mode:
        if selected_mode not in modes_cfg:
            raise ValueError(f"Unknown mode '{selected_mode}'. Available: {', '.join(sorted(modes_cfg.keys()))}")
        mode_cfg = dict(modes_cfg[selected_mode] or {})
        normalized["mode"] = selected_mode
        if not args.study_name and mode_cfg.get("study_name"):
            normalized["study_name"] = str(mode_cfg["study_name"]).strip()
        if not args.output_dir and mode_cfg.get("output_dir"):
            normalized["output_dir"] = resolve_path(str(mode_cfg["output_dir"]), base_dir=PROJECT_ROOT)
            if not storage_value:
                normalized["storage_path"] = normalized["output_dir"] / "study.db"
                normalized["storage"] = f"sqlite:///{normalized['storage_path'].as_posix()}"
        normalized["static_args"] = normalized["static_args"] + [str(item) for item in mode_cfg.get("static_args", [])]
        if "metric" in mode_cfg:
            merged_metric = dict(normalized["metric"])
            merged_metric.update(dict(mode_cfg.get("metric", {})))
            normalized["metric"] = merged_metric
        mode_parameters = dict(mode_cfg.get("parameters", {}))
        if mode_parameters:
            normalized["parameters"].update(mode_parameters)
        parameter_names = mode_cfg.get("parameter_names")
        if parameter_names:
            normalized["parameters"] = {name: normalized["parameters"][name] for name in parameter_names}

    if normalized["direction"] not in {"maximize", "minimize"}:
        raise ValueError("study.direction must be 'maximize' or 'minimize'")
    if normalized["metric"]["type"] not in {"json", "csv"}:
        raise ValueError("metric.type must be 'json' or 'csv'")
    return normalized


def json_scalar(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def sample_parameter(trial: Any, name: str, spec: dict[str, Any]) -> Any:
    suggest = str(spec.get("suggest", spec.get("type", ""))).strip().lower()
    if suggest in {"float", "suggest_float"}:
        return trial.suggest_float(name, float(spec["low"]), float(spec["high"]), log=bool(spec.get("log", False)), step=spec.get("step"))
    if suggest in {"int", "suggest_int"}:
        step = int(spec.get("step", 1))
        return trial.suggest_int(name, int(spec["low"]), int(spec["high"]), step=step, log=bool(spec.get("log", False)))
    if suggest in {"categorical", "choice", "choices", "suggest_categorical"}:
        choices = spec.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError(f"Parameter '{name}' requires a non-empty choices list")
        return trial.suggest_categorical(name, choices)
    raise ValueError(f"Unsupported parameter suggest type for '{name}': {suggest}")


def append_parameter_args(command: list[str], param_name: str, param_value: Any, spec: dict[str, Any]) -> None:
    target = str(spec.get("target", "override")).strip().lower()
    if target == "override":
        command.extend(["--set", f"{param_name}={json_scalar(param_value)}"])
        return

    if target != "arg":
        raise ValueError(f"Unsupported target for '{param_name}': {target}")

    arg_name = str(spec.get("arg", "")).strip()
    if not arg_name:
        raise ValueError(f"Parameter '{param_name}' uses target=arg but no arg field was provided")

    if bool(spec.get("flag", False)):
        if bool(param_value):
            command.append(arg_name)
        return

    command.extend([arg_name, str(param_value)])


def apply_config_parameter(config_payloads: dict[str, dict[str, Any]], param_name: str, param_value: Any, spec: dict[str, Any]) -> None:
    updates = spec.get("config_updates", [])
    if not isinstance(updates, list) or not updates:
        raise ValueError(f"Parameter '{param_name}' uses target=config but has no config_updates")
    for update in updates:
        if not isinstance(update, dict):
            raise ValueError(f"Parameter '{param_name}' has invalid config_updates entry")
        config_name = str(update.get("config", "")).strip()
        dotted_key = str(update.get("key", "")).strip()
        if config_name not in config_payloads:
            raise ValueError(f"Parameter '{param_name}' references unknown runtime config '{config_name}'")
        if not dotted_key:
            raise ValueError(f"Parameter '{param_name}' config update is missing key")
        assign_nested_value(config_payloads[config_name], dotted_key, param_value)


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def build_metric_path(metric_cfg: dict[str, Any], run_output_dir: Path, cwd: Path) -> Path:
    path = Path(metric_cfg["path"])
    if path.is_absolute():
        return path
    return (run_output_dir / path) if run_output_dir else (cwd / path)


def lookup_json_key(payload: dict[str, Any], dotted_key: str) -> Any:
    cursor: Any = payload
    for part in dotted_key.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            raise KeyError(f"Missing metric key: {dotted_key}")
        cursor = cursor[part]
    return cursor


def value_matches(actual: str | None, expected: Any) -> bool:
    if actual is None:
        return False
    if isinstance(expected, (int, float)):
        try:
            return math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-9)
        except ValueError:
            return False
    return str(actual) == str(expected)


def load_metric_value(metric_cfg: dict[str, Any], run_output_dir: Path, cwd: Path) -> float:
    metric_path = build_metric_path(metric_cfg, run_output_dir, cwd)
    if not metric_path.exists():
        raise FileNotFoundError(f"Metric file not found: {metric_path}")

    if metric_cfg["type"] == "json":
        with open(metric_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        metric_value = lookup_json_key(payload, metric_cfg["key"])
        return float(metric_value)

    with open(metric_path, "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    row_filter = metric_cfg.get("row_filter", {})
    selected_rows = rows
    if row_filter:
        selected_rows = [
            row
            for row in rows
            if all(value_matches(row.get(key), expected) for key, expected in row_filter.items())
        ]
    if not selected_rows:
        raise ValueError(f"No matching row found in metric CSV: {metric_path}")
    column = metric_cfg.get("column", "")
    if not column:
        raise ValueError("metric.column must be configured for csv metrics")
    return float(selected_rows[0][column])


def build_trial_command(study_cfg: dict[str, Any], trial: Any, trial_dir: Path) -> tuple[list[str], dict[str, Any], Path | None]:
    command = list(study_cfg["command"])
    command.extend(study_cfg["static_args"])

    run_output_dir: Path | None = None
    if study_cfg["output_arg"]:
        run_output_dir = trial_dir / "run_output"
        run_output_dir.mkdir(parents=True, exist_ok=True)
        command.extend([study_cfg["output_arg"], str(run_output_dir)])

    runtime_config_payloads: dict[str, dict[str, Any]] = {}
    runtime_configs = study_cfg.get("runtime_configs", {})
    train_base = runtime_configs.get("train_base")
    finetune_base = runtime_configs.get("finetune_base")
    if train_base is not None:
        runtime_config_payloads["train"] = copy.deepcopy(load_yaml(train_base))
    if finetune_base is not None:
        runtime_config_payloads["finetune"] = copy.deepcopy(load_yaml(finetune_base))

    sampled_params: dict[str, Any] = {}
    for param_name, raw_spec in study_cfg["parameters"].items():
        if not isinstance(raw_spec, dict):
            raise ValueError(f"Parameter spec for '{param_name}' must be a mapping")
        value = sample_parameter(trial, param_name, raw_spec)
        sampled_params[param_name] = value
        target = str(raw_spec.get("target", "override")).strip().lower()
        if target == "config":
            apply_config_parameter(runtime_config_payloads, param_name, value, raw_spec)
        else:
            append_parameter_args(command, param_name, value, raw_spec)

    if "train" in runtime_config_payloads:
        train_config_path = trial_dir / "runtime_train_config.yaml"
        write_yaml(train_config_path, runtime_config_payloads["train"])
        command.extend([runtime_configs.get("train_arg", "-TrainConfig"), str(train_config_path)])
    if "finetune" in runtime_config_payloads:
        finetune_config_path = trial_dir / "runtime_finetune_config.yaml"
        write_yaml(finetune_config_path, runtime_config_payloads["finetune"])
        command.extend([runtime_configs.get("finetune_arg", "-FinetuneConfig"), str(finetune_config_path)])

    return command, sampled_params, run_output_dir


def timestamp_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def build_sampler(study_cfg: dict[str, Any], optuna_module: Any) -> Any:
    sampler_cfg = study_cfg.get("sampler", {})
    sampler_name = str(sampler_cfg.get("name", "tpe")).strip().lower()
    seed = sampler_cfg.get("seed", 42)
    if sampler_name == "random":
        return optuna_module.samplers.RandomSampler(seed=seed)
    return optuna_module.samplers.TPESampler(
        seed=seed,
        n_startup_trials=int(sampler_cfg.get("n_startup_trials", 5)),
        multivariate=bool(sampler_cfg.get("multivariate", True)),
    )


def summarize_trials(study: Any) -> dict[str, Any]:
    completed = [trial for trial in study.trials if trial.state.name == "COMPLETE" and trial.value is not None]
    failed = [trial for trial in study.trials if trial.state.name == "FAIL"]
    pruned = [trial for trial in study.trials if trial.state.name == "PRUNED"]
    summary: dict[str, Any] = {
        "study_name": study.study_name,
        "direction": str(study.direction).split(".")[-1].lower(),
        "trial_count": len(study.trials),
        "completed_trials": len(completed),
        "failed_trials": len(failed),
        "pruned_trials": len(pruned),
    }
    if completed:
        values = [float(trial.value) for trial in completed]
        summary["metric_stats"] = {
            "best": max(values),
            "worst": min(values),
            "mean": statistics.fmean(values),
            "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        }
    else:
        summary["metric_stats"] = {}
    return summary


def write_study_summaries(study: Any, study_cfg: dict[str, Any]) -> None:
    output_dir = study_cfg["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for trial in study.trials:
        row: dict[str, Any] = {
            "trial_number": trial.number,
            "state": trial.state.name,
            "value": trial.value,
        }
        row.update({f"param.{key}": value for key, value in trial.params.items()})
        for attr_key, attr_value in trial.user_attrs.items():
            if isinstance(attr_value, (dict, list)):
                row[f"attr.{attr_key}"] = json.dumps(attr_value, ensure_ascii=False)
            else:
                row[f"attr.{attr_key}"] = attr_value
        rows.append(row)

    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    csv_path = output_dir / "trials.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize_trials(study)
    best_payload: dict[str, Any]
    try:
        best_trial = study.best_trial
        best_payload = {
            "trial_number": best_trial.number,
            "value": best_trial.value,
            "params": best_trial.params,
            "user_attrs": best_trial.user_attrs,
        }
    except ValueError:
        best_payload = {}

    write_json(output_dir / "study_summary.json", summary)
    write_json(output_dir / "best_trial.json", best_payload)

    md_lines = [
        f"# Optuna Summary: {study.study_name}",
        "",
        f"- Generated at: {timestamp_text()}",
        f"- Direction: {summary['direction']}",
        f"- Total trials: {summary['trial_count']}",
        f"- Completed trials: {summary['completed_trials']}",
        f"- Failed trials: {summary['failed_trials']}",
        f"- Pruned trials: {summary['pruned_trials']}",
        "",
    ]
    if best_payload:
        md_lines.extend(
            [
                "## Best Trial",
                "",
                f"- Trial: {best_payload['trial_number']}",
                f"- Value: {best_payload['value']}",
                f"- Output: {best_payload['user_attrs'].get('run_output_dir', '')}",
                "",
                "## Best Params",
                "",
            ]
        )
        for key, value in best_payload["params"].items():
            md_lines.append(f"- {key}: {value}")
        md_lines.append("")

    top_trials = [trial for trial in study.trials if trial.state.name == "COMPLETE" and trial.value is not None]
    reverse = study_cfg["direction"] == "maximize"
    top_trials.sort(key=lambda item: float(item.value), reverse=reverse)
    if top_trials:
        md_lines.extend([
            "## Top Trials",
            "",
            "| trial | value | state |",
            "| --- | ---: | --- |",
        ])
        for trial in top_trials[:10]:
            md_lines.append(f"| {trial.number} | {float(trial.value):.6f} | {trial.state.name} |")
        md_lines.append("")

    with open(output_dir / "study_summary.md", "w", encoding="utf-8") as handle:
        handle.write("\n".join(md_lines).strip() + "\n")


def run_trial_process(study_cfg: dict[str, Any], trial: Any) -> float:
    trial_dir = study_cfg["output_dir"] / "trials" / f"trial_{trial.number:04d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    command, sampled_params, run_output_dir = build_trial_command(study_cfg, trial, trial_dir)

    stdout_path = trial_dir / "stdout.log"
    stderr_path = trial_dir / "stderr.log"
    command_text = subprocess.list2cmdline(command)
    write_json(
        trial_dir / "trial_plan.json",
        {
            "trial_number": trial.number,
            "created_at": timestamp_text(),
            "cwd": str(study_cfg["cwd"]),
            "command": command,
            "command_text": command_text,
            "params": sampled_params,
            "run_output_dir": str(run_output_dir) if run_output_dir else "",
        },
    )

    with open(stdout_path, "w", encoding="utf-8") as stdout_handle, open(stderr_path, "w", encoding="utf-8") as stderr_handle:
        child_env = dict(os.environ)
        child_env["PYTHON_EXE"] = sys.executable
        completed = subprocess.run(
            command,
            cwd=study_cfg["cwd"],
            env=child_env,
            stdout=stdout_handle,
            stderr=stderr_handle,
            check=False,
            text=True,
        )

    if completed.returncode != 0:
        trial.set_user_attr("command", command_text)
        trial.set_user_attr("stdout_log", str(stdout_path))
        trial.set_user_attr("stderr_log", str(stderr_path))
        raise RuntimeError(f"Trial {trial.number} failed with exit code {completed.returncode}")

    metric_value = load_metric_value(study_cfg["metric"], run_output_dir or study_cfg["output_dir"], study_cfg["cwd"])
    trial.set_user_attr("command", command_text)
    trial.set_user_attr("stdout_log", str(stdout_path))
    trial.set_user_attr("stderr_log", str(stderr_path))
    trial.set_user_attr("run_output_dir", str(run_output_dir) if run_output_dir else "")
    trial.set_user_attr("metric_path", str(build_metric_path(study_cfg["metric"], run_output_dir or study_cfg["output_dir"], study_cfg["cwd"])))
    trial.set_user_attr("params", sampled_params)
    write_json(
        trial_dir / "trial_result.json",
        {
            "trial_number": trial.number,
            "completed_at": timestamp_text(),
            "value": metric_value,
            "params": sampled_params,
            "run_output_dir": str(run_output_dir) if run_output_dir else "",
            "metric_path": str(build_metric_path(study_cfg["metric"], run_output_dir or study_cfg["output_dir"], study_cfg["cwd"])),
        },
    )
    return metric_value


def import_optuna() -> Any:
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError("Optuna is not installed. Run 'pip install -r requirements.txt' first.") from exc
    return optuna


def main() -> None:
    args = parse_args()
    raw_cfg = load_yaml(Path(args.study_config))
    study_cfg = normalize_study_config(raw_cfg, args, Path(args.study_config))
    study_cfg["output_dir"].mkdir(parents=True, exist_ok=True)

    optuna = import_optuna()
    sampler = build_sampler(study_cfg, optuna)
    study = optuna.create_study(
        study_name=study_cfg["study_name"],
        storage=study_cfg["storage"],
        direction=study_cfg["direction"],
        sampler=sampler,
        load_if_exists=True,
    )

    if args.summary_only:
        write_study_summaries(study, study_cfg)
        print(f"Summary refreshed under: {study_cfg['output_dir']}")
        return

    catch_exceptions: tuple[type[BaseException], ...] = () if args.fail_fast else (RuntimeError, FileNotFoundError, ValueError)
    study.optimize(
        lambda trial: run_trial_process(study_cfg, trial),
        n_trials=study_cfg["n_trials"],
        timeout=study_cfg["timeout"],
        catch=catch_exceptions,
    )
    write_study_summaries(study, study_cfg)

    try:
        best_trial = study.best_trial
        print(f"Best trial: {best_trial.number}")
        print(f"Best value: {best_trial.value}")
        print(f"Summary dir: {study_cfg['output_dir']}")
    except ValueError:
        print(f"No completed trial was produced. See: {study_cfg['output_dir']}")


if __name__ == "__main__":
    main()