"""Launcher to run grid-world RL experiments in parallel and plot combined results.

This script spawns subprocesses running `2.py` with different `--algo` flags, waits for them to finish,
then loads each `metrics_<algo>.npz` file and plots training (episode) rewards and evaluation means on one figure.
"""
import os
import sys
import subprocess
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
SCRIPT = os.path.join(HERE, '2.py')
OUT = os.path.join(HERE, 'metrics_grid')
os.makedirs(OUT, exist_ok=True)

ALGOS = ['PPO', 'SAC', 'GRPO']
# include TRPO if available (2.py will skip if not)
ALGOS.append('TRPO')

procs = []
for algo in ALGOS:
    cmd = [sys.executable, SCRIPT, '--algo', algo, '--timesteps', '20000', '--out-dir', OUT, '--seed', '0']
    print('Launching', ' '.join(cmd))
    f = open(os.path.join(OUT, f'log_{algo.lower()}.txt'), 'w')
    p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
    procs.append({'proc': p, 'file': f, 'algo': algo})

# wait with progress bar
try:
    from tqdm import tqdm
    use_tqdm = True
except Exception:
    use_tqdm = False

if use_tqdm:
    with tqdm(total=len(procs), desc='Completed') as pbar:
        alive = procs.copy()
        while alive:
            for entry in alive[:]:
                p = entry['proc']
                if p.poll() is not None:
                    # finished
                    entry['file'].close()
                    print('Finished', entry['algo'])
                    alive.remove(entry)
                    pbar.update(1)
            time.sleep(0.2)
else:
    for entry in procs:
        p = entry['proc']
        p.wait()
        entry['file'].close()
        print('Finished', entry['algo'])

# collect and plot
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
for algo in ALGOS:
    path = os.path.join(OUT, f'metrics_{algo.lower()}.npz')
    if not os.path.exists(path):
        print('Missing', path)
        continue
    d = np.load(path)
    tr = d.get('train_rewards')
    if tr is not None and len(tr) > 0:
        plt.plot(np.arange(len(tr)), tr, label=f'{algo} train')
plt.xlabel('episode')
plt.ylabel('train episode reward')
plt.legend()

plt.subplot(1,2,2)
labels = []
means = []
errs = []
for algo in ALGOS:
    path = os.path.join(OUT, f'metrics_{algo.lower()}.npz')
    if not os.path.exists(path):
        continue
    d = np.load(path)
    means.append(float(d['mean']))
    errs.append(float(d['std']))
    labels.append(algo)
x = np.arange(len(labels))
plt.bar(x, means, yerr=errs, capsize=5)
plt.xticks(x, labels)
plt.ylabel('mean eval reward')
plt.title('Eval comparison')

plt.tight_layout()
out = os.path.join(OUT, 'comparison_parallel.png')
plt.savefig(out)
print('Saved', out)
