"""
Load a YAML config and merge with CLI overrides into an argparse.Namespace
that KERMT's existing code (modify_train_args, KermtFinetuneTask, etc.) can
consume directly.

Priority (highest → lowest):
  1. Explicit CLI flags  (--epochs 20)
  2. YAML config file    (epochs: 10)
  3. argparse defaults   (epochs: 30)
"""

import sys
import os
import copy
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Optional, Dict, Any

import yaml


# ── boolean flags that argparse treats as store_true ──
_STORE_TRUE_FLAGS = frozenset({
    "no_cuda", "no_cache", "no_features_scaling", "self_attention",
    "tensorboard", "features_only", "use_compound_names",
    "save_smiles_splits", "show_individual_scores",
    "distinct_init", "select_by_loss", "enbl_multi_gpu",
    "use_cuikmolmaker_featurization",
})


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


def yaml_to_cli_args(cfg: Dict[str, Any]) -> list:
    """Convert a flat YAML dict into a sys.argv-style list."""
    tokens = []
    for key, value in cfg.items():
        flag = f"--{key}"
        if key in _STORE_TRUE_FLAGS:
            if value:
                tokens.append(flag)
        elif isinstance(value, list):
            tokens.append(flag)
            tokens.extend(str(v) for v in value)
        else:
            tokens.append(flag)
            tokens.append(str(value))
    return tokens


def build_args(
    config_path: str,
    cli_overrides: Optional[list] = None,
    subcommand: str = "finetune",
) -> Namespace:
    """
    Build a fully resolved Namespace by:
      1. Parsing YAML into baseline args
      2. Overlaying explicit CLI overrides
      3. Running KERMT's modify_train_args / modify_predict_args

    Returns the final Namespace ready for cross_validate() or make_predictions().
    """
    kermt_root = Path(__file__).resolve().parents[2]
    if str(kermt_root) not in sys.path:
        sys.path.insert(0, str(kermt_root))

    from kermt.util.parsing import (
        add_finetune_args,
        add_predict_args,
        modify_train_args,
        modify_predict_args,
    )

    cfg = load_yaml(config_path)

    if cli_overrides:
        override_cfg = _parse_cli_overrides(cli_overrides, cfg)
        cfg.update(override_cfg)

    _TLC_EXTRA_KEYS = {
        "solvent_emb_dim", "n_solvent_dims", "n_cross_dims", "early_stop_epoch",
        "regression_loss",
    }
    extra_cfg = {k: cfg.pop(k) for k in list(cfg) if k in _TLC_EXTRA_KEYS}

    # argparse 对 --no_cache 是 store_true 且默认 True，YAML 写 no_cache: false 时不会生成 flag，
    # 需在 parse 后显式关掉，否则无法启用 MolGraph 缓存。
    explicit_no_cache = cfg.get("no_cache")

    yaml_tokens = yaml_to_cli_args(cfg)

    parser = ArgumentParser()
    if subcommand in ("finetune", "eval"):
        add_finetune_args(parser)
    elif subcommand == "predict":
        add_predict_args(parser)
    else:
        raise ValueError(f"Unsupported subcommand: {subcommand}")

    args = parser.parse_args(yaml_tokens)
    args.parser_name = subcommand

    for k, v in extra_cfg.items():
        setattr(args, k, v)

    if explicit_no_cache is False:
        args.no_cache = False

    if subcommand in ("finetune", "eval"):
        modify_train_args(args)
    elif subcommand == "predict":
        modify_predict_args(args)

    return args


def _parse_cli_overrides(tokens: list, base_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse raw CLI tokens like ['--epochs', '20', '--no_cuda'] into a dict
    that can overlay the YAML config.
    """
    overrides = {}
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if not token.startswith("--"):
            i += 1
            continue
        key = token.lstrip("-")

        if key == "config":
            i += 2
            continue

        if key in _STORE_TRUE_FLAGS:
            if i + 1 < len(tokens) and tokens[i + 1].lower() in (
                "true", "false", "1", "0", "yes", "no",
            ):
                overrides[key] = tokens[i + 1].lower() in ("true", "1", "yes")
                i += 2
            else:
                overrides[key] = True
                i += 1
            continue

        if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
            raw = tokens[i + 1]
            overrides[key] = _cast_value(raw, base_cfg.get(key))
            i += 2
        else:
            overrides[key] = True
            i += 1

    return overrides


def _cast_value(raw: str, reference):
    """Cast a string CLI value to match the type in the YAML config."""
    if reference is None:
        return _auto_cast(raw)
    if isinstance(reference, bool):
        return raw.lower() in ("true", "1", "yes")
    if isinstance(reference, int):
        return int(raw)
    if isinstance(reference, float):
        return float(raw)
    return raw


def _auto_cast(raw: str):
    """Best-effort cast when no YAML reference exists."""
    for caster in (int, float):
        try:
            return caster(raw)
        except (ValueError, TypeError):
            continue
    if raw.lower() in ("true", "false"):
        return raw.lower() == "true"
    return raw


def resolve_paths(args: Namespace, base_dir: str):
    """Make relative paths in args absolute, anchored to base_dir."""
    path_keys = [
        "data_path", "separate_val_path", "separate_test_path",
        "checkpoint_path", "checkpoint_dir", "save_dir",
        "features_path", "separate_val_features_path",
        "separate_test_features_path", "output_path",
    ]
    for key in path_keys:
        val = getattr(args, key, None)
        if val is None:
            continue
        if isinstance(val, list):
            setattr(args, key, [
                os.path.join(base_dir, v) if not os.path.isabs(v) else v
                for v in val
            ])
        elif isinstance(val, str) and not os.path.isabs(val):
            setattr(args, key, os.path.join(base_dir, val))
