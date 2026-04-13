from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


COMMANDS = {
    "phase1": [sys.executable, str(ROOT / "scripts" / "train_phase1_eegnet.py")],
    "phase1-baseline": [sys.executable, str(ROOT / "scripts" / "baseline_phase1_cpp_features.py")],
    "phase2-cue": [sys.executable, str(ROOT / "scripts" / "train_phase2_cue_dimensionality.py")],
    "phase2-rt": [sys.executable, str(ROOT / "scripts" / "train_phase2_rt_fastslow.py")],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Master script for reproducible EEG experiment entry points.")
    parser.add_argument("target", choices=COMMANDS.keys())
    parser.add_argument("args", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    command = COMMANDS[args.target] + args.args
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
