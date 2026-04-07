"""Mini project: GridWorld + PPO / SAC / TRPO / GRPO comparisons

This script implements:
- a very small continuous GridWorld environment (Gymnasium API)
- training and evaluation harness using Stable-Baselines3 for PPO and SAC
- optional TRPO if `sb3_contrib` is installed
- a simple GRPO surrogate (PPO with optimizer weight decay as gradient regularization)

Notes / limitations:
- SAC requires a continuous action space, so the environment is continuous (agent moves in continuous 2D).
- TRPO is attempted via `sb3_contrib.trpo.TRPO`; if that package is not installed the script will skip TRPO and continue.
- GRPO here is implemented as PPO whose optimizer uses weight decay (a simple proxy for gradient regularization).

Usage: run the script (no CLI required). It will train each algorithm for a small number of timesteps, evaluate them,
save per-algorithm metrics under `./metrics_grid/` and write a comparison plot `comparison_grid.png`.

Dependencies (install if missing):
  pip install gymnasium stable-baselines3[extra] torch numpy matplotlib sb3-contrib
"""

import os
import math
import time
import random
from typing import Tuple, Dict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
	import gymnasium as gym
except Exception as e:
	raise RuntimeError('Please install gymnasium: pip install gymnasium') from e

try:
	from stable_baselines3 import PPO, SAC
except Exception as e:
	raise RuntimeError('Please install stable-baselines3 (pip install stable-baselines3[extra])') from e

# TRPO from sb3_contrib (optional)
try:
	from sb3_contrib.trpo import TRPO
	HAVE_TRPO = True
except Exception:
	TRPO = None
	HAVE_TRPO = False

import torch
import torch as th
import torch.nn as nn


class ContinuousGridWorld(gym.Env):
	"""Simple continuous 2D grid world.

	State: agent position (x,y) with values in [0, size-1].
	Action: continuous 2-vector in [-1,1] representing delta movement; actual move = action * step_size.
	Reward: +1 when agent is within goal_radius of goal; episode ends. Small step penalty optional.
	Observation: normalized (x/(size-1), y/(size-1)).
	"""

	metadata = {"render_modes": ["human"]}

	def __init__(self, size: int = 5, step_size: float = 1.0, max_steps: int = 50, goal_radius: float = 0.5):
		super().__init__()
		self.size = size
		self.step_size = step_size
		self.max_steps = max_steps
		self.goal_radius = goal_radius

		# continuous 2D action
		self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
		# normalized position observation
		self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)

		self._rng = np.random.RandomState(0)
		self.reset()

	def reset(self, seed: int = None, options: dict = None):
		if seed is not None:
			self._rng = np.random.RandomState(seed)
		# start at near (0,0)
		self.pos = np.array([0.0, 0.0], dtype=np.float32)
		# fixed goal at far corner
		self.goal = np.array([float(self.size - 1), float(self.size - 1)], dtype=np.float32)
		self.steps = 0
		obs = self._get_obs()
		return obs, {}

	def _get_obs(self):
		return np.array([self.pos[0] / (self.size - 1), self.pos[1] / (self.size - 1)], dtype=np.float32)

	def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
		action = np.clip(action, -1.0, 1.0)
		delta = action * self.step_size
		self.pos = np.clip(self.pos + delta, 0.0, float(self.size - 1))
		self.steps += 1
		dist = np.linalg.norm(self.pos - self.goal)
		done = False
		reward = 0.0
		if dist <= self.goal_radius:
			reward = 1.0
			done = True
		# small time penalty to encourage shorter trajectories (optional)
		reward -= 0.0
		truncated = self.steps >= self.max_steps
		if truncated:
			done = True
		return self._get_obs(), float(reward), bool(done), bool(truncated), {}

	def render(self, mode: str = "human"):
		grid = np.zeros((self.size, self.size), dtype=str)
		grid[:] = '.'
		ax = int(round(self.pos[0]))
		ay = int(round(self.pos[1]))
		gx = int(self.goal[0])
		gy = int(self.goal[1])
		grid[ay, ax] = 'A'
		grid[gy, gx] = 'G'
		print('\n'.join(''.join(row) for row in grid))


def make_env(size=5, max_steps=50):
	return ContinuousGridWorld(size=size, max_steps=max_steps)


def evaluate_model(model, env, n_episodes=20) -> Dict[str, float]:
	rewards = []
	for _ in range(n_episodes):
		obs, _ = env.reset()
		done = False
		total = 0.0
		truncated = False
		while not done and not truncated:
			action, _ = model.predict(obs, deterministic=True)
			obs, r, done, truncated, _ = env.step(action)
			total += r
		rewards.append(total)
	return {"mean": float(np.mean(rewards)), "std": float(np.std(rewards)), "all": np.array(rewards)}


def run_one(algo: str, out_dir: str = './metrics_grid', timesteps: int = 20000, seed: int = 42):
	"""Run a single algorithm and save its monitor/training rewards and evaluation results."""
	os.makedirs(out_dir, exist_ok=True)
	train_env = make_env(size=5, max_steps=50)
	eval_env = make_env(size=5, max_steps=50)
	from stable_baselines3.common.monitor import Monitor

	monitor_file = os.path.join(out_dir, f"monitor_{algo.lower()}.csv")
	train_env = Monitor(train_env, filename=monitor_file)

	np.random.seed(seed)
	random.seed(seed)
	th.manual_seed(seed)

	model = None
	if algo == 'PPO':
		model = PPO('MlpPolicy', train_env, verbose=1, seed=seed)
		model.learn(total_timesteps=timesteps)
	elif algo == 'SAC':
		model = SAC('MlpPolicy', train_env, verbose=1, seed=seed)
		model.learn(total_timesteps=timesteps)
	elif algo == 'TRPO' and HAVE_TRPO:
		model = TRPO('MlpPolicy', train_env, verbose=1, seed=seed)
		model.learn(total_timesteps=timesteps)
	elif algo == 'GRPO':
		model = PPO('MlpPolicy', train_env, verbose=1, seed=seed)
		try:
			base_lr = model.lr_schedule(1)
		except Exception:
			base_lr = 3e-4
		params = model.policy.parameters()
		model.optimizer = th.optim.Adam(params, lr=base_lr, weight_decay=1e-4)
		model.learn(total_timesteps=timesteps)
	else:
		raise ValueError(f'Unknown/unsupported algo {algo}')

	# save model
	try:
		model.save(os.path.join(out_dir, f"{algo.lower()}_model.zip"))
	except Exception:
		pass

	# read monitor CSV to collect training episode rewards
	train_rewards = []
	if os.path.exists(monitor_file):
		with open(monitor_file, 'r') as f:
			for line in f:
				if line.startswith('#'):
					continue
				parts = line.strip().split(',')
				if len(parts) >= 3:
					# Monitor writes total reward in first column 'r'
					try:
						r = float(parts[0])
						train_rewards.append(r)
					except Exception:
						continue

	# evaluate
	eval_res = evaluate_model(model, eval_env, n_episodes=50)

	# save metrics
	np.savez(os.path.join(out_dir, f"metrics_{algo.lower()}.npz"), mean=eval_res['mean'], std=eval_res['std'], eval_rewards=eval_res['all'], train_rewards=np.array(train_rewards))
	print(f"Saved metrics for {algo} to {out_dir}")
	return eval_res


def run_experiments(out_dir: str = './metrics_grid', timesteps: int = 20000, seed: int = 42):
	algos = ['PPO', 'SAC']
	if HAVE_TRPO:
		algos.append('TRPO')
	else:
		print('sb3_contrib.TRPO not available; TRPO will be skipped')
	algos.append('GRPO')

	results = {}
	for algo in algos:
		results[algo] = run_one(algo, out_dir=out_dir, timesteps=timesteps, seed=seed)

	# plot summary (evaluation means and training reward curves available per-algo files)
	# this function keeps the old behavior but uses per-algo saved npz files
	labels = []
	means = []
	errs = []
	for k, v in results.items():
		labels.append(k)
		means.append(v['mean'])
		errs.append(v['std'])

	plt.figure(figsize=(6,4))
	x = np.arange(len(labels))
	plt.bar(x, means, yerr=errs, capsize=5, color=['tab:blue','tab:orange','tab:green','tab:purple'][:len(labels)])
	plt.xticks(x, labels)
	plt.ylabel('Mean episode reward')
	plt.title('Algorithm comparison on ContinuousGridWorld')
	outpng = os.path.join(out_dir, 'comparison_grid.png')
	plt.tight_layout()
	plt.savefig(outpng)
	print('Saved comparison plot to', outpng)


if __name__ == '__main__':
	import argparse
	p = argparse.ArgumentParser()
	p.add_argument('--algo', type=str, default=None, help='Run single algorithm (PPO,SAC,TRPO,GRPO)')
	p.add_argument('--timesteps', type=int, default=20000)
	p.add_argument('--out-dir', type=str, default='./metrics_grid')
	p.add_argument('--seed', type=int, default=0)
	args = p.parse_args()
	if args.algo is None:
		run_experiments(out_dir=args.out_dir, timesteps=args.timesteps, seed=args.seed)
	else:
		run_one(args.algo, out_dir=args.out_dir, timesteps=args.timesteps, seed=args.seed)