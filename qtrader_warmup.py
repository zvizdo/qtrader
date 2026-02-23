"""
Standalone CLI to pre-compute and cache all state provider outputs for the full date range.

Usage:
    python qtrader_warmup.py
    python qtrader_warmup.py --date-start 2016-01-01 --date-end 2026-02-01
    python qtrader_warmup.py --golden-db /path/to/golden_cache.db
"""

import os
import shutil
import pathlib
import argparse

import click
import json
from datetime import datetime

from lean.commands.backtest import backtest as run_lean_backtest
from qtrader.rlflow.persistence import PersistenceJSONEncoder


WARMUP_NAME = "warmup-cache"
WARMUP_RUN = "warmup"


def run_warmup(date_start, date_end, golden_db):
    proj_path = pathlib.Path(__file__).parent.resolve()

    # Paths mirror the trainer convention:
    #   storage_dir  = ../storage/{name}          (top-level trial storage)
    #   config_dir   = ../storage/{name}/{run}    (where params.json lives)
    #   output_dir   = ../trials/{name}           (Lean backtest output)
    storage_dir = proj_path.joinpath(f"../storage/{WARMUP_NAME}").resolve()
    config_dir = storage_dir.joinpath(WARMUP_RUN)
    output_dir = proj_path.joinpath(f"../trials/{WARMUP_NAME}").resolve()

    os.makedirs(config_dir, exist_ok=True)

    # Write params for main.py to pick up via object_store.get_file_path("{name}/params.json")
    # Lean sets self.name = backtest_name = "{WARMUP_NAME}/{WARMUP_RUN}"
    # main.py reads: self.object_store.get_file_path(f"{self.name}/params.json")
    #   → resolves to ../storage/{WARMUP_NAME}/{WARMUP_RUN}/params.json
    run_params = {
        "run_type": "WARMUP",
        "hyperparams": {},
        "date_start": date_start.isoformat(),
        "date_end": date_end.isoformat(),
    }
    params_path = config_dir.joinpath("params.json")
    with open(params_path, "wb") as f:
        f.write(json.dumps(run_params, cls=PersistenceJSONEncoder).encode())

    # Run the Lean backtest in WARMUP mode
    print(f"Running WARMUP backtest: {date_start} → {date_end}")
    ctx = click.Context(run_lean_backtest)
    ctx.forward(
        run_lean_backtest,
        backtest_name=f"{WARMUP_NAME}/{WARMUP_RUN}",
        project=proj_path,
        output=output_dir,
        detach=False,
        no_update=True,
    )

    # The golden cache is the db.sqlite produced by the WARMUP run.
    # Lean's object store writes it at ../storage/{WARMUP_NAME}/db.sqlite
    src_db = storage_dir.joinpath("db.sqlite")
    if not src_db.exists():
        print(f"ERROR: Expected db.sqlite not found at {src_db}")
        return

    golden_db = pathlib.Path(golden_db).resolve()
    os.makedirs(golden_db.parent, exist_ok=True)
    shutil.copy2(src_db, golden_db)
    print(f"Golden cache DB written to: {golden_db}")

    # Clean up the entire warmup storage and trial output
    if storage_dir.exists():
        shutil.rmtree(storage_dir)
        print(f"Cleaned up warmup storage: {storage_dir}")

    if output_dir.exists():
        shutil.rmtree(output_dir)
        print(f"Cleaned up warmup output: {output_dir}")

    print("WARMUP complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-compute state cache for all training date ranges."
    )
    parser.add_argument(
        "--date-start",
        type=str,
        default="2016-01-01",
        help="Start date (YYYY-MM-DD). Default: 2016-01-01",
    )
    parser.add_argument(
        "--date-end",
        type=str,
        default="2026-02-01",
        help="End date (YYYY-MM-DD). Default: 2026-02-01",
    )
    parser.add_argument(
        "--golden-db",
        type=str,
        default=None,
        help="Path for the golden cache DB. Default: ../storage/cache/golden_cache.db",
    )

    args = parser.parse_args()
    date_start = datetime.fromisoformat(args.date_start)
    date_end = datetime.fromisoformat(args.date_end)

    if args.golden_db is None:
        proj_path = pathlib.Path(__file__).parent.resolve()
        golden_db = proj_path.joinpath("../storage/cache/golden_cache.db").resolve()
    else:
        golden_db = args.golden_db

    run_warmup(date_start, date_end, golden_db)
