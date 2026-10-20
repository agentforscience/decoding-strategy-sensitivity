"""GPU scheduler for the shared host: launches one generate.py process per model on whichever
GPU currently has enough free memory, retrying models whose launch fails (e.g. startup OOM
caused by other users' jobs). Logs to logs/gen_<model>.log and logs/run_all.status.
"""
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import MODELS, ROOT, TASKS, gen_path, runs  # noqa: E402

MAX_TRIES = 30
LOG = os.path.join(ROOT, "logs")


def free_mem():
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return {int(a): int(b) / 1024 for a, b in (l.split(",") for l in out.strip().splitlines())}


def complete(m):
    return all(os.path.exists(gen_path(m, t, s, sd)) for t in TASKS for s, sd in runs())


def is_running(m):
    """True if a generate.py process for model m is alive (launched by us or manually)."""
    busy = subprocess.run(["pgrep", "-af", "generate.py --model"], capture_output=True, text=True).stdout
    return any(l.split("--model ")[1].split()[0] == m for l in busy.splitlines() if "--model " in l)


def status(msg):
    with open(os.path.join(LOG, "run_all.status"), "a") as f:
        f.write(time.strftime("%H:%M:%S ") + msg + "\n")


def main():
    models = sys.argv[1].split(",") if len(sys.argv) > 1 else list(MODELS)
    pending = [m for m in models if not complete(m) and not is_running(m)]
    tries = {m: 0 for m in pending}
    running = {}  # model -> (Popen, gpu)
    while pending or running:
        for m, (p, g) in list(running.items()):
            if p.poll() is not None:
                del running[m]
                if complete(m):
                    status(f"done {m}")
                else:
                    tries[m] += 1
                    status(f"fail {m} rc={p.returncode} try={tries[m]}")
                    if tries[m] < MAX_TRIES:
                        pending.append(m)
        if pending:
            fm = free_mem()
            # Launch the first pending model (in priority order) that fits on any GPU.
            for m in list(pending):
                need = 0.8 * MODELS[m][1] + 0.5
                g = max(fm, key=fm.get)
                if is_running(m) or complete(m):
                    pending.remove(m)
                    continue
                if fm[g] >= need:
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
                    lf = open(os.path.join(LOG, f"gen_{m}.log"), "a")
                    p = subprocess.Popen([sys.executable, os.path.join(ROOT, "src", "generate.py"), "--model", m],
                                         env=env, stdout=lf, stderr=subprocess.STDOUT)
                    running[m] = (p, g)
                    pending.remove(m)
                    status(f"launch {m} on gpu{g} (free {fm[g]:.1f}GB)")
                    time.sleep(75)  # let the process grab its memory before measuring again
                    break
        time.sleep(10)
    status("ALL FINISHED")


if __name__ == "__main__":
    main()
