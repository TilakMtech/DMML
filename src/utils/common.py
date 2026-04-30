from __future__ import annotations
import json, logging, os, random
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]

def load_config(path: str | Path = 'configs/config.yaml') -> dict:
    with open(ROOT / path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def ensure_dirs(*paths: str | Path) -> None:
    for p in paths:
        (ROOT / p if not Path(p).is_absolute() else Path(p)).mkdir(parents=True, exist_ok=True)

def utc_now() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

def run_partition() -> str:
    return datetime.now(timezone.utc).strftime('dt=%Y-%m-%d/hour=%H')

def setup_logger(name: str, log_file: str | Path) -> logging.Logger:
    ensure_dirs(Path(log_file).parent)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s')
    fh = logging.FileHandler(ROOT / log_file, encoding='utf-8')
    sh = logging.StreamHandler()
    fh.setFormatter(fmt); sh.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(sh)
    return logger

def save_json(obj: dict, path: str | Path) -> None:
    full = ROOT / path
    full.parent.mkdir(parents=True, exist_ok=True)
    with open(full, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, default=str)

def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed)
