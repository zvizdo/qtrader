import argparse
import subprocess
import uuid
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-jobs", help="Trials to run in parallel", type=int, default=2)

    parser.add_argument("--name", help="Study name", type=str, default='study-test')
    parser.add_argument("--num-trials", help="Number of trials to run", type=int, default=10)
    parser.add_argument("--path", help="Study path", type=str, default='trial.db')

    args = parser.parse_args()
    print(args)
    n_jobs = args.n_jobs
    study_name = args.name
    num_trials = args.num_trials
    path = args.path

    trials_ran = 0
    trials = []

    while trials_ran < num_trials:
        for i, tpack in enumerate(trials):
            pid, trial, pid_logs = tpack
            if trial.poll() is not None:
                pid_logs.close()
                del trials[i]
                print(f"RUNNER: Trial finished! Return code: {trial.poll()}; PID: {pid}")

        for n in range(len(trials), n_jobs):
            pid = uuid.uuid4().hex
            pid_logs = open(f'./backtests/logs/{pid}.txt', 'w+')
            t = subprocess.Popen(
                ["python", "qtrader_optimize.py",
                 "--name", study_name,
                 "--path", path,
                 "--num-trials", "1"],
                stdout=pid_logs,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
            trials.append((pid, t, pid_logs))
            trials_ran += 1
            print(f"RUNNER: New trial ran; Total: {trials_ran}")

        time.sleep(5)
