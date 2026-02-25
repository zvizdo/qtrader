import numpy as np

np.random.seed(42)

import time
import os, shutil, pathlib
import argparse

import click
import json
import optuna
from lean.commands.backtest import backtest as run_lean_backtest
from lean.commands.report import report as run_report
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

from qtrader.rlflow.persistence import SQLitePersistenceProvider, PersistenceJSONEncoder
from qtrader.logging import TrainingLogger


TR_INFO__STEP = "TRAINING_INFO__STEP"


class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split("=")
            getattr(namespace, self.dest)[key] = value


def dump_params(config_dir_f, run_params, run_name=""):
    os.makedirs(config_dir_f(run_name), exist_ok=True)
    with open(config_dir_f(run_name).joinpath("params.json"), "wb") as f:
        f.write(json.dumps(run_params, cls=PersistenceJSONEncoder).encode())


def get_run_stats(output_path: pathlib.Path):
    backtest_id = None
    with open(output_path.joinpath("config")) as f:
        backtest_id = json.load(f)["id"]

    with open(output_path.joinpath(f"{backtest_id}.json")) as f:
        return backtest_id, json.load(f)


def run_backtest(
    proj_path,
    config_dir_f,
    iter_dir_f,
    name,
    run_name,
    run_params,
    delete_lean_folder=True,
):
    ctx = click.Context(run_lean_backtest)
    dump_params(config_dir_f, run_params, run_name=run_name)

    output_path = iter_dir_f(run_name)
    last_exc = None
    for t in np.random.randint(0, 16, size=5):
        time.sleep(t)
        try:
            ctx.forward(
                run_lean_backtest,
                backtest_name=f"{name}/{run_name}",
                project=proj_path,
                output=output_path,
                detach=False,
                no_update=True,
                # extra_config=[("log-level", "error")]
            )
            last_exc = None
            break

        except Exception as e:
            print(f"BACKTEST RETRY FAILED [{run_name}]: {e}")
            last_exc = e

    if last_exc is not None:
        raise RuntimeError(
            f"All backtest attempts failed for '{run_name}'"
        ) from last_exc

    id, s = get_run_stats(output_path)

    if not delete_lean_folder:  # keep the backtest folder and add report
        try:
            bt_dst = output_path.joinpath(
                f"../../../qtrader/backtests/{name}-{run_name}"
            ).resolve()
            shutil.copytree(src=output_path, dst=bt_dst)
            ctx.forward(
                run_report,
                backtest_results=bt_dst.joinpath(f"{id}.json"),
                report_destination=output_path.joinpath(f"readout.html"),
                detach=False,
            )

        except Exception as e:
            print(f"DOCKER FAIL: lean report > {e}")

        finally:
            shutil.rmtree(bt_dst)

    else:  # delete backtest folder
        shutil.rmtree(output_path)

    shutil.rmtree(config_dir_f(run_name))
    return s


def trainer_run(name, iters, params, n_test=5, prune=None, golden_db=None):
    proj_path = pathlib.Path(__file__).parent.resolve()
    trial_dir = proj_path.joinpath(f"../trials/{name}").resolve()
    iter_dir_f = lambda x: trial_dir.joinpath(f"{x}").resolve()
    config_dir_f = lambda x: proj_path.joinpath(f"../storage/{name}/{x}").resolve()

    # --- Training and Evaluation Period Config ---
    train_period_start = datetime(2016, 1, 1)
    train_period_end = datetime(2023, 1, 1) # Approximate 7 years from start
    train_sample_duration_days = 35

    eval_period_start = datetime(2023, 2, 1)
    eval_period_end = datetime(2026, 1, 31)
    eval_sample_duration_days = 70

    # Golden cache DB: pre-computed state provider outputs
    if golden_db is None:
        golden_db = proj_path.joinpath("../storage/cache/golden_cache.db").resolve()
    else:
        golden_db = pathlib.Path(golden_db).resolve()

    if not golden_db.exists():
        print(f"WARNING: Golden cache DB not found at {golden_db}. "
              "State providers will compute from scratch each iteration. "
              "Run qtrader_warmup.py to create it.")

    run_params = {"run_type": "EVAL", "hyperparams": params}

    os.makedirs(config_dir_f(""), exist_ok=True)
    # Copy golden cache DB into storage folder (once, before training loop)
    if golden_db.exists():
        dst = config_dir_f("").joinpath("db.sqlite")
        print (f"Copying golden cache DB from {golden_db} to {dst}")
        if not dst.exists():
            shutil.copy2(golden_db, dst)
            print(f"Golden cache DB copied to {dst}")

    pprovider = SQLitePersistenceProvider(root=config_dir_f(""))
    dump_params(config_dir_f, run_params)

    logger = TrainingLogger(
        log_dir=str(proj_path.joinpath(f"../trials/{'tfboard'}/{name}").resolve())
    )
    logger.log_hyperparams(params)

    print(f"START [{name}]: {params}")
    step = 0
    try:
        step = pprovider.load_dict(name=TR_INFO__STEP)
    except:
        pprovider.persist_dict(name=TR_INFO__STEP, obj=step)

    eval_stats = []
    for i in range(iters):
        # region Run Iteration
        step += 1
        pprovider.persist_dict(name=TR_INFO__STEP, obj=step)
        print(f"ITERATION: {i + 1}/{iters}")

        iter_name = f"Iter{str(step).zfill(6)}"
        run_params["run_type"] = "TRAIN"
        run_params["date_start"] = train_period_start + timedelta(
            days=np.random.randint(0, (train_period_end - train_period_start).days - train_sample_duration_days)
        )
        run_params["date_end"] = run_params["date_start"] + timedelta(days=train_sample_duration_days)

        stats = run_backtest(
            proj_path, config_dir_f, iter_dir_f, name, iter_name, run_params, True
        )
        logger.log_train_step(step=step, stats=stats)
        # endregion

        if (i + 1) % n_test > 0 and (i + 1) != iters:
            continue

        # region Run Eval
        eval_name = f"Eval{str(step).zfill(6)}"

        eval_start = eval_period_start + timedelta(
            days=np.random.randint(0, (eval_period_end - eval_period_start).days - eval_sample_duration_days)
        )
        eval_end = eval_start + timedelta(days=eval_sample_duration_days)

        run_params.update(
            {
                "run_type": "EVAL",
                "date_start": eval_start,
                "date_end": eval_end,
            }
        )
        stats = run_backtest(
            proj_path, config_dir_f, iter_dir_f, name, eval_name, run_params, False
        )
        eval_stats.append(stats)
        logger.log_eval_step(step=step, stats=stats)
        # endregion

        # Multi-metric hard prune (only when running inside Optuna)
        if prune is not None and step >= 200:
            port_stats = stats.get("totalPerformance", {}).get("portfolioStatistics", {})
            sharpe = float(port_stats.get("sharpeRatio", 0))
            drawdown = float(port_stats.get("drawdown", 0))
            num_trades = len(stats.get("orders", {}).keys())

            if num_trades <= 10 or drawdown > 0.5 or sharpe < -1.0:
                raise optuna.TrialPruned()

        # Percentile-based pruning via Optuna callback
        if prune is not None and callable(prune):
            prune(
                float(stats["totalPerformance"]["portfolioStatistics"]["sharpeRatio"]),
                step,
            )

    logger.close()
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", help="Name", type=str, default="Study-DQN-Tr001")
    parser.add_argument("--iters", help="Iterations", type=int, default=25)
    parser.add_argument("-p", "--params", nargs="*", action=ParseKwargs)

    args, args_unkwn = parser.parse_known_args()
    print(args)
    name = args.name
    iters = args.iters
    params = args.params if args.params is not None else {}

    trainer_run(name, iters, params, n_test=10)

