"""Shared locations independent of the process working directory."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
DATA = ROOT / "data"
CONFIGS = ROOT / "configs"
RUNS = ROOT / "runs"
