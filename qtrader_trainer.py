import numpy as np

np.random.seed(42)

import time
import os, shutil, pathlib
import argparse

import click
import json
import optuna
import tensorflow as tf
from lean.commands.backtest import backtest as run_lean_backtest
from lean.commands.report import report as run_report
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

from qtrader.rlflow.persistence import SQLitePersistenceProvider, PersistenceJSONEncoder


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
    # Fix #9: log exceptions on each retry; raise if all attempts fail
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


def tfb_record(writer, step, mode, stats):
    ttl_perf = stats["totalPerformance"]

    pnl = lambda s: float(s["portfolioStatistics"]["endEquity"]) - float(
        s["portfolioStatistics"]["startEquity"]
    )
    num_trades = lambda s: int(s["tradeStatistics"]["totalNumberOfTrades"])
    num_orders = lambda s: len(s["orders"].keys())
    sh_ratio = lambda s: float(s["portfolioStatistics"]["sharpeRatio"])
    so_ratio = lambda s: float(s["portfolioStatistics"]["sortinoRatio"])

    with writer.as_default(step=step):
        if mode == "TRAIN":
            if "mean_td_error" in stats["statistics"]:
                tf.summary.scalar(
                    f"{mode}/mean_td_error", float(stats["statistics"]["mean_td_error"])
                )

            if "mean_reward" in stats["statistics"]:
                tf.summary.scalar(
                    f"{mode}/mean_reward", float(stats["statistics"]["mean_reward"])
                )

        elif mode == "EVAL":
            tf.summary.scalar(f"{mode}/profit", pnl(ttl_perf))
            tf.summary.scalar(
                f"{mode}/sharpe_ratio",
                sh_ratio(ttl_perf),
            )
            tf.summary.scalar(f"{mode}/sortino_ratio", so_ratio(ttl_perf))
            tf.summary.scalar(f"{mode}/num_trades", num_trades(ttl_perf))
            tf.summary.scalar(f"{mode}/num_orders", num_orders(stats))
        writer.flush()


def trainer_run(name, iters, params, n_test=5, prune=None):
    proj_path = pathlib.Path(__file__).parent.resolve()
    trial_dir = proj_path.joinpath(f"../trials/{name}").resolve()
    iter_dir_f = lambda x: trial_dir.joinpath(f"{x}").resolve()
    config_dir_f = lambda x: proj_path.joinpath(f"../storage/{name}/{x}").resolve()

    run_params = {"run_type": "EVAL", "hyperparams": params}
    data_start_date = datetime(2016, 1, 1)

    os.makedirs(config_dir_f(""), exist_ok=True)
    pprovider = SQLitePersistenceProvider(root=config_dir_f(""))
    dump_params(config_dir_f, run_params)

    tfb_writer = tf.summary.create_file_writer(
        logdir=str(proj_path.joinpath(f"../trials/{'tfboard'}/{name}").resolve())
    )

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
        run_params["date_start"] = data_start_date + timedelta(
            days=np.random.randint(0, 365 * 2)
        )
        run_params["date_end"] = run_params["date_start"] + relativedelta(years=3)

        stats = run_backtest(
            proj_path, config_dir_f, iter_dir_f, name, iter_name, run_params, True
        )
        tfb_record(tfb_writer, step=step, mode=run_params["run_type"], stats=stats)
        # endregion

        if (i + 1) % n_test > 0 and (i + 1) != iters:
            continue

        # region Run Eval
        eval_name = f"Eval{str(step).zfill(6)}"
        run_params.update(
            {
                "run_type": "EVAL",
                "date_start": datetime(2021, 1, 1),
                "date_end": datetime(2024, 12, 31),
            }
        )
        stats = run_backtest(
            proj_path, config_dir_f, iter_dir_f, name, eval_name, run_params, False
        )
        eval_stats.append(stats)
        tfb_record(tfb_writer, step=step, mode=run_params["run_type"], stats=stats)
        # endregion

        # Fix #8: only raise TrialPruned when running inside Optuna (prune is not None)
        num_orders = lambda s: len(s["orders"].keys())
        if prune is not None and step >= 50 and num_orders(stats) <= 15:
            raise optuna.TrialPruned()

        if prune is not None and callable(prune):
            prune(
                float(stats["totalPerformance"]["portfolioStatistics"]["sharpeRatio"]),
                step,
            )

    tfb_writer.close()
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
