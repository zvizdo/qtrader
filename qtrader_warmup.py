"""
Standalone CLI to pre-compute and cache all state provider outputs for the full date range in parallel.

Usage:
    python qtrader_warmup.py
    python qtrader_warmup.py --date-start 2016-01-01 --date-end 2026-02-01 --workers 6
    python qtrader_warmup.py --golden-db /path/to/golden_cache.db --chunk-months 4
    python qtrader_warmup.py --date-start 2016-01-01 --date-end 2025-01-01 --workers 6 --chunk-months 1
"""

import os
import sys
import shutil
import pathlib
import argparse
import json
import sqlite3
import contextlib
from datetime import datetime

import click
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from lean.commands.backtest import backtest as run_lean_backtest
from qtrader.rlflow.persistence import PersistenceJSONEncoder


WARMUP_NAME = "warmup-cache"
WARMUP_RUN = "warmup"


def add_months(d, months):
    new_month = d.month - 1 + months
    year = d.year + new_month // 12
    month = new_month % 12 + 1
    # Handle end of month issues, e.g. Jan 31 + 1 month = Feb 28
    day = min(d.day, [31, 29 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month - 1])
    return d.replace(year=year, month=month, day=day)


def generate_chunks(date_start, date_end, chunk_months):
    chunks = []
    current_start = date_start
    while current_start < date_end:
        current_end = add_months(current_start, chunk_months)
        if current_end > date_end:
            current_end = date_end
        chunks.append((current_start, current_end))
        current_start = current_end
    return chunks


def run_warmup_chunk(chunk_start, chunk_end, chunk_id):
    proj_path = pathlib.Path(__file__).parent.resolve()

    chunk_name = f"{WARMUP_NAME}-chunk-{chunk_id}"
    storage_dir = proj_path.joinpath(f"../storage/{chunk_name}").resolve()
    config_dir = storage_dir.joinpath(WARMUP_RUN)
    output_dir = proj_path.joinpath(f"../trials/{chunk_name}").resolve()

    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Write params for main.py to pick up via object_store.get_file_path("{name}/params.json")
    run_params = {
        "run_type": "WARMUP",
        "hyperparams": {},
        "date_start": chunk_start.isoformat(),
        "date_end": chunk_end.isoformat(),
    }
    params_path = config_dir.joinpath("params.json")
    with open(params_path, "wb") as f:
        f.write(json.dumps(run_params, cls=PersistenceJSONEncoder).encode())

    # The log path isolates stdout/stderr for each chunk to avoid polluting the terminal
    log_path = output_dir.joinpath("chunk.log")

    try:
        with open(log_path, "w") as log_f, contextlib.redirect_stdout(log_f), contextlib.redirect_stderr(log_f):
            # Run the Lean backtest in WARMUP mode isolated
            ctx = click.Context(run_lean_backtest)
            ctx.forward(
                run_lean_backtest,
                backtest_name=f"{chunk_name}/{WARMUP_RUN}",
                project=proj_path,
                output=output_dir,
                detach=False,
                no_update=True,
                download_data=False,
                data_provider_historical="Local"
            )

        src_db = storage_dir.joinpath("db.sqlite")
        if not src_db.exists():
            return chunk_id, False, f"Expected db.sqlite not found at {src_db}. Check log at {log_path}", None, None

        return chunk_id, True, "Success", src_db, (storage_dir, output_dir)
    except Exception as e:
        return chunk_id, False, str(e), None, None


def merge_db(src_db_path, dest_db_path):
    dest_conn = sqlite3.connect(dest_db_path, timeout=60.0)
    dest_conn.execute("PRAGMA journal_mode=WAL;")
    dest_conn.execute("PRAGMA synchronous=NORMAL;")

    dest_conn.execute(f"ATTACH DATABASE '{src_db_path}' AS src;")

    dest_conn.execute(
        """
        CREATE TABLE IF NOT EXISTS data (
            id TEXT PRIMARY KEY,
            payload BLOB
        );
        """
    )

    dest_conn.execute("INSERT OR REPLACE INTO data SELECT * FROM src.data;")
    dest_conn.commit()

    dest_conn.execute("DETACH DATABASE src;")
    dest_conn.close()


def run_warmup_parallel(date_start, date_end, golden_db, chunk_months, workers):
    golden_db = pathlib.Path(golden_db).resolve()
    os.makedirs(golden_db.parent, exist_ok=True)

    chunks = generate_chunks(date_start, date_end, chunk_months)
    print(f"Divided date range into {len(chunks)} chunks.")
    print(f"Using {workers} workers.")

    # Sort chunks so progress feels more linear chronologically (optional, but nice)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(run_warmup_chunk, chunk[0], chunk[1], i): i
            for i, chunk in enumerate(chunks)
        }

        with tqdm(total=len(chunks), desc="Processing chunks") as pbar:
            for fut in as_completed(futures):
                chunk_id = futures[fut]
                try:
                    c_id, success, msg, src_db, dirs_to_clean = fut.result()
                    if success:
                        merge_db(src_db, golden_db)
                        # Clean up the entire warmup storage and trial output for this chunk
                        storage_dir, output_dir = dirs_to_clean
                        if storage_dir.exists():
                            shutil.rmtree(storage_dir, ignore_errors=True)
                        if output_dir.exists():
                            shutil.rmtree(output_dir, ignore_errors=True)
                    else:
                        print(f"\nChunk {c_id} failed: {msg}")
                except Exception as e:
                    print(f"\nChunk {chunk_id} raised an exception: {e}")
                finally:
                    pbar.update(1)

    print(f"WARMUP complete. Golden cache DB written/merged at: {golden_db}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-compute state cache for all training date ranges in parallel."
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
    parser.add_argument(
        "--chunk-months",
        type=int,
        default=4,
        help="Number of months per chunk. Default: 4",
    )
    
    # max(1, os.cpu_count() - 2) logic
    cpu_cores = os.cpu_count() or 4
    default_workers = max(1, cpu_cores - 2)
    parser.add_argument(
        "--workers",
        type=int,
        default=default_workers,
        help=f"Number of parallel workers. Default: {default_workers}",
    )

    args = parser.parse_args()
    date_start = datetime.fromisoformat(args.date_start)
    date_end = datetime.fromisoformat(args.date_end)

    if args.golden_db is None:
        proj_path = pathlib.Path(__file__).parent.resolve()
        golden_db = proj_path.joinpath("../storage/cache/golden_cache.db").resolve()
    else:
        golden_db = args.golden_db

    run_warmup_parallel(date_start, date_end, golden_db, args.chunk_months, args.workers)
