import argparse
import os
import subprocess
import uuid
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-jobs", help="Trials to run in parallel", type=int, default=2)

    parser.add_argument("--name", help="Study name", type=str, default='study-test')
    parser.add_argument("--num-trials", help="Number of trials to run", type=int, default=10)
    parser.add_argument("--path", help="Study path", type=str, default='trial.db')
    parser.add_argument("--iters", help="Training iterations per trial", type=int, default=150)

    args = parser.parse_args()
    print(args)
    n_jobs = args.n_jobs
    study_name = args.name
    num_trials = args.num_trials
    path = args.path
    iters = args.iters

    # Fix #4: ensure the log directory exists before writing
    os.makedirs('./backtests/logs', exist_ok=True)

    trials_ran = 0
    trials = []

    while trials_ran < num_trials:
        # Fix #1: iterate over a snapshot to avoid mutating the list mid-loop;
        # rebuild trials in-place to only keep still-running processes.
        still_running = []
        for tpack in trials:
            pid, trial, pid_logs = tpack
            rc = trial.poll()  # Fix #2: capture return code once
            if rc is not None:
                pid_logs.close()
                print(f"RUNNER: Trial finished! Return code: {rc}; PID: {pid}")
            else:
                still_running.append(tpack)
        trials = still_running

        for n in range(len(trials), n_jobs):
            # Fix #3: do not spawn more than the requested total
            if trials_ran >= num_trials:
                break
            pid = uuid.uuid4().hex
            pid_logs = open(f'./backtests/logs/{pid}.txt', 'w+')
            t = subprocess.Popen(
                ["python", "qtrader_optimize.py",
                 "--name", study_name,
                 "--path", path,
                 "--num-trials", "1",
                 "--iters", str(iters)],
                stdout=pid_logs,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
            trials.append((pid, t, pid_logs))
            trials_ran += 1
            print(f"RUNNER: New trial ran; Total: {trials_ran}")

        time.sleep(5)
