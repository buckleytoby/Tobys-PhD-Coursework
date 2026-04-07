# existing file
"""
Simple MNIST transformer training script.

Features:
- Loads MNIST (optionally subset for quick tests)
- Small patch-based transformer classifier
- Two optimizers: Adam and a Hessian-diagonal Newton-like optimizer using Hutchinson estimator
- Training loop that switches between optimizers every N steps

Usage: run the script directly. See argparse flags for options.
"""

import math
import sys
from typing import List, Union

import torch
import os
import numpy as np
import subprocess
import time
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset


class PatchEmbedding(nn.Module):
	def __init__(self, in_channels: int, patch_size: int, emb_dim: int, img_size: int = 28):
		super().__init__()
		self.patch_size = patch_size
		self.num_patches = (img_size // patch_size) ** 2
		self.proj = nn.Conv2d(in_channels, emb_dim, kernel_size=patch_size, stride=patch_size)

	def forward(self, x):
		# x: (B, C, H, W)
		x = self.proj(x)  # (B, emb_dim, H/ps, W/ps)
		x = x.flatten(2).transpose(1, 2)  # (B, num_patches, emb_dim)
		return x


class TransformerClassifier(nn.Module):
	def __init__(self, *, img_size=28, patch_size=7, in_channels=1, emb_dim=64, n_heads=4, n_layers=2, mlp_dim=128, n_classes=10, dropout=0.1):
		super().__init__()
		self.patch_embed = PatchEmbedding(in_channels, patch_size, emb_dim, img_size=img_size)
		num_patches = self.patch_embed.num_patches
		self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
		self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, emb_dim))
		encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=n_heads, dim_feedforward=mlp_dim, dropout=dropout, activation='gelu')
		self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
		self.norm = nn.LayerNorm(emb_dim)
		self.head = nn.Linear(emb_dim, n_classes)

	def forward(self, x):
		# x: (B, 1, 28, 28)
		B = x.shape[0]
		x = self.patch_embed(x)  # (B, P, D)
		cls_tokens = self.cls_token.expand(B, -1, -1)
		x = torch.cat([cls_tokens, x], dim=1)  # (B, P+1, D)
		x = x + self.pos_embed
		# transformer expects (S, B, D)
		x = x.transpose(0, 1)
		x = self.transformer(x)
		x = x.transpose(0, 1)
		cls_out = x[:, 0]
		cls_out = self.norm(cls_out)
		logits = self.head(cls_out)
		return logits


class HillClassifier(nn.Module):
	"""
	Hill-like classifier: repeatedly expand by outer product with a learned vector,
	flatten, pass through linear projection, repeat, then final classifier.
	This is intentionally simple and cheap for experimentation.
	"""
	def __init__(self, in_dim=28*28, inner_dim=16, stages=2, n_classes=10, ks=None):
		super().__init__()
		self.in_dim = in_dim
		self.inner_dim = inner_dim
		self.stages = stages
		# allow ks to be a scalar (repeat) or list of ints per stage
		if ks is None:
			ks = [inner_dim] * stages
		elif isinstance(ks, int):
			ks = [ks] * stages
		assert len(ks) == stages, 'ks must have length == stages'
		self.ks = ks
		# per-stage vector to form outer product (each has length ks[i])
		self.vecs = nn.ParameterList([nn.Parameter(torch.randn(k)) for k in ks])
		# compute flattened sizes after each expansion
		flat_sizes = []
		cur = in_dim
		for k in ks:
			cur = cur * k
			flat_sizes.append(cur)
		# safety check to avoid enormous intermediate tensors
		for sz in flat_sizes:
			if sz > MAX_FLAT_SIZE:
				raise ValueError(f"Hill expansion too large: flat size {sz} > MAX_FLAT_SIZE={MAX_FLAT_SIZE}. Reduce ks or stages.")
		# linear mapping after flatten: project from the current representation size (rep)
		# down to the target flat size for that stage. During top-down distillation
		# rep starts at flat_sizes[-1] and after each step becomes flat_sizes[i], so
		# projector[i] should accept input size = flat_sizes[i+1] (or flat_sizes[i] for last)
		self.projectors = nn.ModuleList()
		for i in range(self.stages):
			if i + 1 < len(flat_sizes):
				in_size = flat_sizes[i+1]
			else:
				in_size = flat_sizes[i]
			out_size = flat_sizes[i]
			self.projectors.append(nn.Linear(in_size, out_size))
		# learned per-stage distillation weights sized to flat sizes
		self.distill_weights = nn.ParameterList([nn.Parameter(torch.ones(flat_sizes[i])) for i in range(stages)])
		# final reducer to map flattened outer0 back to in_dim
		self.final_reducer = nn.Linear(flat_sizes[0], in_dim)
		self.head = nn.Linear(in_dim, n_classes)

	def forward(self, x):
		# x shape: (B, 1, 28, 28)
		B = x.shape[0]
		x = x.view(B, -1)  # (B, in_dim)
		# First: compute all expansions (outer products) and store flattened tensors and outers
		flats = []
		outers = []
		x_cur = x
		for i in range(self.stages):
			v = self.vecs[i]
			# outer: (B, L_prev, k_i)
			outer = x_cur.unsqueeze(2) * v.unsqueeze(0).unsqueeze(0)
			outers.append(outer)
			flat = outer.view(B, -1)  # (B, L_prev * k_i)
			flats.append(flat)
			# next stage input is this flat
			x_cur = flat

		# Second: distill top-down: for each stage add projected reshaped matrix to corresponding outer
		rep = flats[-1]
		for i in range(self.stages - 1, -1, -1):
			proj_flat = self.projectors[i](rep)  # (B, flat_sizes[i])
			# apply learned per-element distillation weight
			w = self.distill_weights[i]
			proj_flat = proj_flat * w.unsqueeze(0)
			proj_flat = F.relu(proj_flat)
			# reshape back to outer shape
			k = self.ks[i]
			L_prev = proj_flat.shape[1] // k
			proj_outer = proj_flat.view(B, L_prev, k)
			# add to corresponding outer
			outers[i] = outers[i] + proj_outer
			# next rep is flattened updated outer
			rep = outers[i].view(B, -1)

		# reduce final flattened outer0 back to in_dim and residual-add to x
		reduced = self.final_reducer(rep)
		x = x + reduced
		logits = self.head(x)
		return logits


class ShallowWideMLP(nn.Module):
	"""Single hidden layer MLP (very wide)"""
	def __init__(self, in_dim=28*28, hidden=4660, n_classes=10):
		super().__init__()
		self.fc1 = nn.Linear(in_dim, hidden)
		self.act = nn.ReLU()
		self.fc2 = nn.Linear(hidden, n_classes)

	def forward(self, x):
		B = x.shape[0]
		x = x.view(B, -1)
		x = self.act(self.fc1(x))
		return self.fc2(x)


class NarrowDeepMLP(nn.Module):
	"""Deep MLP with many narrow layers"""
	def __init__(self, in_dim=28*28, hidden=673, layers=8, n_classes=10):
		super().__init__()
		self.layers = nn.ModuleList()
		# first layer
		self.layers.append(nn.Linear(in_dim, hidden))
		for _ in range(layers - 1):
			self.layers.append(nn.Linear(hidden, hidden))
		self.head = nn.Linear(hidden, n_classes)
		self.act = nn.ReLU()

	def forward(self, x):
		B = x.shape[0]
		x = x.view(B, -1)
		for l in self.layers:
			x = self.act(l(x))
		return self.head(x)

class SimpleCNN(nn.Module):
	"""Small CNN for MNIST classification."""
	def __init__(self, in_channels=1, n_classes=10):
		super().__init__()
		# two conv blocks with pooling
		self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
		self.bn1 = nn.BatchNorm2d(32)
		self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
		self.bn2 = nn.BatchNorm2d(64)
		self.pool = nn.MaxPool2d(2)
		# after two pools: 28 -> 14 -> 7, feature map size = 7
		self.flatten_dim = 64 * 7 * 7
		self.fc1 = nn.Linear(self.flatten_dim, 512)
		self.act = nn.ReLU()
		self.head = nn.Linear(512, n_classes)

	def forward(self, x):
		# x: (B,1,28,28)
		x = self.pool(self.act(self.bn1(self.conv1(x))))
		x = self.pool(self.act(self.bn2(self.conv2(x))))
		B = x.shape[0]
		x = x.view(B, -1)
		x = self.act(self.fc1(x))
		return self.head(x)


def hutchinson_diag_estimator(loss, params: List[torch.nn.Parameter], num_samples: int = 1, device: Union[torch.device, str] = 'cpu'):
	"""
	Estimate diagonal of the Hessian using the Hutchinson estimator.

	Returns a list of tensors matching params shapes with diagonal estimates.
	"""
	# normalize device to torch.device
	# accept device as string or torch.device
	dev = torch.device(device)
	# First-order gradients
	grads = torch.autograd.grad(loss, params, create_graph=True)
	diag_est = [torch.zeros_like(p, device=dev) for p in params]
	for _ in range(num_samples):
		# Rademacher probe vectors (+1 or -1)
		probes = [torch.randint(0, 2, p.shape, device=dev).float() * 2.0 - 1.0 for p in params]
		# compute g^T v
		gv = torch.zeros((), device=dev)
		for g, v in zip(grads, probes):
			gv = gv + (g * v).sum()
		# hv = grad(gv, params)
		hv = torch.autograd.grad(gv, params, retain_graph=True)
		# elementwise product hv * v gives diagonal estimator contribution
		for i, (hhi, v, hv_i) in enumerate(zip(diag_est, probes, hv)):
			diag_est[i] = hhi + (hv_i * v)
	diag_est = [d / float(max(1, num_samples)) for d in diag_est]
	# detach to avoid keeping graph
	diag_est = [d.detach() for d in diag_est]
	return diag_est


class HessianDiagOptimizer:
	"""
	Simple Hessian-diagonal Newton-like optimizer using Hutchinson estimator for diag(H).

	This is a lightweight optimizer intended for experimentation and small models.
	It performs parameter-wise updates: p <- p - lr * g / (|h_diag| + damping)
	"""

	def __init__(self, params, lr=1e-2, damping=1e-3, hutchinson_samples=1, device: Union[torch.device, str] = 'cpu'):
		self.params = list(params)
		self.lr = lr
		self.damping = damping
		self.hutchinson_samples = hutchinson_samples
		# accept either string or torch.device
		self.device = torch.device(device)

	def zero_grad(self):
		for p in self.params:
			if p.grad is not None:
				p.grad.detach_()
				p.grad.zero_()

	def step(self, loss):
		# compute grads
		params = [p for p in self.params if p.requires_grad]
		grads = torch.autograd.grad(loss, params, create_graph=True)
		# estimate diagonal Hessian
		h_diag = hutchinson_diag_estimator(loss, params, num_samples=self.hutchinson_samples, device=self.device)
		# apply update
		with torch.no_grad():
			for p, g, h in zip(params, grads, h_diag):
				# use absolute diag to avoid direction flip, add damping
				denom = h.abs() + self.damping
				step = self.lr * (g.detach() / denom)
				p.add_(-step)


	class MLPClassifier(nn.Module):
		"""Simple fully-connected MLP classifier operating on flattened MNIST images."""
		def __init__(self, in_dim=28*28, hidden=None, n_classes=10):
			super().__init__()
			# avoid referencing MLP_HIDDEN at import/definition time; resolve at init
			if hidden is None:
				hidden = globals().get('MLP_HIDDEN', 1024)
			self.fc1 = nn.Linear(in_dim, hidden)
			self.act = nn.ReLU()
			self.fc2 = nn.Linear(hidden, hidden)
			self.fc3 = nn.Linear(hidden, n_classes)

		def forward(self, x):
			B = x.shape[0]
			x = x.view(B, -1)
			x = self.act(self.fc1(x))
			x = self.act(self.fc2(x))
			x = self.fc3(x)
			return x


def train(args):
	device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
	torch.manual_seed(args.seed)

	transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
	train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

	if args.subset is not None and args.subset > 0:
		train_dataset = Subset(train_dataset, list(range(min(len(train_dataset), args.subset))))

	train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=DATA_LOADER_WORKERS, pin_memory=True)

	if ARCH == 'transformer':
		# enlarged transformer to match Hill parameter budget (~3.7M params)
		# emb_dim chosen divisible by n_heads
		model = TransformerClassifier(img_size=28, patch_size=7, emb_dim=276, n_heads=4, n_layers=4, mlp_dim=1104)
	elif ARCH == 'hill':
		model = HillClassifier(in_dim=28*28, inner_dim=HILL_INNER_DIM, stages=HILL_STAGES)
	elif ARCH == 'mlp_shallow':
		# choose hidden size so params ~3.7M: in_dim*hidden + hidden*n_classes ~ 784*H
		model = ShallowWideMLP(in_dim=28*28, hidden=4660, n_classes=10)
	elif ARCH == 'mlp_deep':
		# narrow deep MLP with many layers
		model = NarrowDeepMLP(in_dim=28*28, hidden=673, layers=8, n_classes=10)
	elif ARCH == 'cnn':
		model = SimpleCNN(in_channels=1, n_classes=10)
	else:
		raise ValueError(f'Unknown ARCH: {ARCH}')
	model.to(device)
	# print model size summary for quick inspection
	tot_params = sum(p.numel() for p in model.parameters())
	trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print(f"Model params total={tot_params}, trainable={trainable_params}, approx_mem_MB={tot_params*4/1024**2:.2f}")
	# architecture-specific info
	if hasattr(model, 'ks'):
		# Hill model: report flat sizes and per-batch memory
		ks_local = list(model.ks)
		cur = model.in_dim
		flat_sizes = []
		for k in ks_local:
			cur = cur * k
			flat_sizes.append(cur)
		if flat_sizes:
			max_flat = max(flat_sizes)
			print(f"Hill ks={ks_local}, flat_sizes={flat_sizes}, largest_flat={max_flat}")
			print(f"Per-batch largest flat mem (float32) with batch_size={args.batch_size}: {max_flat * args.batch_size * 4 / 1024**2:.2f} MB")
	if hasattr(model, 'patch_embed'):
		emb = model.patch_embed.proj.out_channels
		num_patches = model.patch_embed.num_patches
		print(f"Transformer emb_dim={emb}, num_patches={num_patches}")

	# create optimizers
	adam_opt = torch.optim.Adam(model.parameters(), lr=args.lr_adam)
	hess_opt = HessianDiagOptimizer(model.parameters(), lr=args.lr_hessian, damping=args.hessian_damping, hutchinson_samples=args.hutchinson_samples, device=device)

	criterion = nn.CrossEntropyLoss()

	active = 'adam' if args.start_with_adam else 'hessian'
	step = 0
	print(f"Starting training on device={device}. start_with_adam={args.start_with_adam}, switch_every={args.switch_every}")
	# allow overrides from args
	adam_on = getattr(args, 'adam_on', ADAM_ON)
	hess_on = getattr(args, 'hess_on', HESS_ON)
	run_tag = getattr(args, 'tag', RUN_TAG)
	print(f"ADAM_ON={adam_on} HESS_ON={hess_on} RUN_TAG={run_tag}")

	# storage for metrics
	steps_list = []
	loss_list = []
	acc_list = []
	# record which optimizer actually executed updates per step (0=none,1=adam,2=hessian)
	actual_used_list = []
	for epoch in range(1, args.epochs + 1):
		model.train()
		running_loss = 0.0
		total = 0
		correct = 0
		# last-used optimizer marker for prints
		actual_used = 'none'
		for batch_idx, (data, target) in enumerate(train_loader):
			data = data.to(device)
			target = target.to(device)
			step += 1

			# choose which optimizer to use this step (respecting global toggles)
			# actual_used records which optimizer executed the parameter update
			actual_used = 'none'
			if active == 'adam' and adam_on:
				adam_opt.zero_grad()
				logits = model(data)
				loss = criterion(logits, target)
				loss.backward()
				adam_opt.step()
				actual_used = 'adam'
			elif active == 'hessian' and hess_on:
				# Hessian optimizer needs a fresh backward graph for grads and Hessian estimation
				hess_opt.zero_grad()
				logits = model(data)
				loss = criterion(logits, target)
				# use step(loss) which computes grads and updates params
				hess_opt.step(loss)
				actual_used = 'hessian'
			else:
				# If the chosen optimizer is turned off, fall back to the other if available,
				# otherwise perform a forward only (no update) to record metrics.
				if active == 'adam' and hess_on:
					hess_opt.zero_grad()
					logits = model(data)
					loss = criterion(logits, target)
					hess_opt.step(loss)
					actual_used = 'hessian'
				elif active == 'hessian' and adam_on:
					adam_opt.zero_grad()
					logits = model(data)
					loss = criterion(logits, target)
					loss.backward()
					adam_opt.step()
					actual_used = 'adam'
				else:
					logits = model(data)
					loss = criterion(logits, target)

			running_loss += float(loss.item()) * data.size(0)
			preds = logits.argmax(dim=1)
			total += data.size(0)
			correct += int((preds == target).sum().item())

			# record per-step aggregated metrics
			steps_list.append(step)
			loss_list.append(float(loss.item()))
			acc_list.append(100.0 * int((preds == target).sum().item()) / float(data.size(0)))
			# also optionally record which optimizer actually updated parameters
			# (we will not explode metrics size; store as small integer: 0=none,1=adam,2=hessian)
			# append numeric code for which optimizer actually updated this step
			actual_used_list.append(0 if actual_used == 'none' else (1 if actual_used == 'adam' else 2))

			# switch optimizer periodically
			if args.switch_every > 0 and (step % args.switch_every == 0):
				active = 'hessian' if active == 'adam' else 'adam'
				print(f"[step {step}] switching optimizer -> {active}")

			if batch_idx % args.log_interval == 0:
				avg_loss = running_loss / max(1, total)
				acc = 100.0 * correct / max(1, total)
				print(f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] step={step} loss={avg_loss:.4f} acc={acc:.2f}% active={active}")

		epoch_loss = running_loss / max(1, total)
		epoch_acc = 100.0 * correct / max(1, total)
		print(f"End epoch {epoch}: loss={epoch_loss:.4f}, acc={epoch_acc:.2f}% used_last={actual_used}")

	# save metrics
	out_path = os.path.join(METRICS_DIR, f"metrics_{run_tag}.npz")
	# include actual_used_list if it exists
	if 'actual_used_list' in locals():
		np.savez(out_path, steps=np.array(steps_list), loss=np.array(loss_list), acc=np.array(acc_list), used=np.array(actual_used_list))
	else:
		np.savez(out_path, steps=np.array(steps_list), loss=np.array(loss_list), acc=np.array(acc_list))
	print(f"Saved metrics to {out_path}")


def launch_and_plot():
	here = os.path.dirname(__file__)
	script = os.path.join(here, os.path.basename(__file__))
	plotscript = os.path.join(here, 'plot_results.py')
	cases = [
		("adam_off", {'ADAM_ON': '0', 'HESS_ON': '1'}),
		("hess_off", {'ADAM_ON': '1', 'HESS_ON': '0'}),
		("both_on", {'ADAM_ON': '1', 'HESS_ON': '1'}),
	]
	procs = []
	log_files = []
	for tag, envvars in cases:
		# build CLI args for child (no external config files)
		adam_flag = envvars['ADAM_ON']
		hess_flag = envvars['HESS_ON']
		log_path = os.path.join(METRICS_DIR, f"log_{tag}.txt")
		f = open(log_path, 'w')
		log_files.append(f)
		# pass all relevant globals as CLI args so child can override
		cmd = [
			sys.executable, script, '--child',
			'--adam-on', adam_flag,
			'--hess-on', hess_flag,
			'--tag', tag,
			'--epochs', str(EPOCHS),
			'--batch-size', str(BATCH_SIZE),
			'--lr-adam', str(LR_ADAM),
			'--lr-hessian', str(LR_HESSIAN),
			'--hessian-damping', str(HESSIAN_DAMPING),
			'--hutchinson-samples', str(HUTCHINSON_SAMPLES),
			'--switch-every', str(SWITCH_EVERY),
			'--start-with-adam', str(int(START_WITH_ADAM)),
			'--subset', str(SUBSET if SUBSET is not None else 0),
			'--seed', str(SEED),
			'--log-interval', str(LOG_INTERVAL),
		]
		print(f"Launching {tag} -> log {log_path}")
		p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
		procs.append((p, f))

	# wait for processes
	try:
		for p, f in procs:
			p.wait()
			f.close()
			print(f"Process {p.pid} finished")
	except KeyboardInterrupt:
		print("KeyboardInterrupt: terminating children")
		for p, f in procs:
			p.terminate()
			f.close()

	# run plotting
	print("Running plotting")
	try:
		plot_metrics()
	except Exception as e:
		print('Plotting failed:', e)
		print('Attempting to run embedded csv exporter')
		export_metrics_csv()


def plot_metrics():
	if not PLOT_BLOCK:
		import matplotlib
		matplotlib.use('Agg')
	import matplotlib.pyplot as plt
	# order legend as: hess on, adam on, both on
	CASES = [
		('hess_off', 'hess on', 'tab:orange', 's'),
		('adam_off', 'adam on', 'tab:blue', 'o'),
		('both_on', 'both on', 'tab:green', '^'),
	]
	plt.figure(figsize=(10,4))
	plt.subplot(1,2,1)
	for tag, label, color, marker in CASES:
		p = os.path.join(METRICS_DIR, f'metrics_{tag}.npz')
		if not os.path.exists(p):
			print('Missing', p)
			continue
		print('Plotting', p)
		d = np.load(p)
		plt.plot(d['steps'], d['loss'], label=label, color=color, marker=marker, linewidth=2, markersize=6)
	plt.xlabel('step')
	plt.ylabel('loss')
	plt.legend()
	plt.title('Loss over time')

	plt.subplot(1,2,2)
	for tag, label, color, marker in CASES:
		p = os.path.join(METRICS_DIR, f'metrics_{tag}.npz')
		if not os.path.exists(p):
			continue
		d = np.load(p)
		plt.plot(d['steps'], d['acc'], label=label, color=color, marker=marker, linewidth=2, markersize=6)
	plt.xlabel('step')
	plt.ylabel('accuracy (%)')
	plt.legend()
	plt.title('Accuracy per-batch')

	out = os.path.join(METRICS_DIR, 'comparison.png')
	plt.tight_layout()
	plt.savefig(out)
	print('Saved comparison plot to', out)
	if PLOT_BLOCK:
		plt.show(block=True)


def export_metrics_csv():
	CASES = ['adam_off', 'hess_off', 'both_on']
	for tag in CASES:
		p = os.path.join(METRICS_DIR, f'metrics_{tag}.npz')
		if not os.path.exists(p):
			continue
		d = np.load(p)
		outcsv = os.path.join(METRICS_DIR, f'metrics_{tag}.csv')
		arr = np.vstack([d['steps'], d['loss'], d['acc']]).T
		np.savetxt(outcsv, arr, delimiter=',', header='step,loss,acc', comments='')
		print('Saved CSV', outcsv)



# --- Top-level configuration globals (edit these instead of using CLI) ---
EPOCHS = 5
BATCH_SIZE = 128
LR_ADAM = 1e-3
LR_HESSIAN = 1e-2
HESSIAN_DAMPING = 1e-2
HUTCHINSON_SAMPLES = 1
SWITCH_EVERY = 200  # 0 = never
START_WITH_ADAM = True
SUBSET = 1024  # set to None or 0 to use full dataset
SEED = 42
FORCE_CPU = False
LOG_INTERVAL = 50
# safer defaults to avoid huge memory/compute during experiments
DATA_LOADER_WORKERS = 0  # 0 is safest on macOS
MAX_FLAT_SIZE = 500_000  # max allowed flattened size during hill expansions

# runtime toggles (edit these defaults at top of script if you want different behavior)
ADAM_ON = True
HESS_ON = True
# tag for output files (child runs will override)
RUN_TAG = 'run'
# directory for metrics
METRICS_DIR = os.path.join(os.path.dirname(__file__), 'metrics')
os.makedirs(METRICS_DIR, exist_ok=True)

# If True, keep plot window open (calls plt.show(block=True)).
# If False (default), use non-interactive backend and only save PNG (no pop-up).
PLOT_BLOCK = True

# If True, this process will act as a launcher: spawn 3 subprocesses running this
# script with different optimizer toggles, wait for them, then run the plotter.
LAUNCH_PARALLEL = False

# Architecture selection: 'transformer' or 'hill'
ARCH = 'hill'
# Hill architecture hyperparams — reduced inner dim to keep parameter counts comparable
HILL_STAGES = 1
HILL_INNER_DIM = 2
# MLP hyperparams (target ~3.7M params)
MLP_HIDDEN = 4660


if __name__ == '__main__':
	# build a simple args-like namespace from globals for train()
	from types import SimpleNamespace
	# simple CLI parsing: support '--child' and '--launch'. Child accepts many flags.
	is_launch = '--launch' in sys.argv
	is_child = '--child' in sys.argv

	# defaults from globals
	child_adam = ADAM_ON
	child_hess = HESS_ON
	child_tag = RUN_TAG
	child_epochs = EPOCHS
	child_batch = BATCH_SIZE
	child_lr_adam = LR_ADAM
	child_lr_hessian = LR_HESSIAN
	child_hessian_damping = HESSIAN_DAMPING
	child_hutchinson_samples = HUTCHINSON_SAMPLES
	child_switch_every = SWITCH_EVERY
	child_start_with_adam = START_WITH_ADAM
	child_subset = SUBSET
	child_seed = SEED
	child_log_interval = LOG_INTERVAL

	if is_child:
		argv = sys.argv[1:]
		i = 0
		while i < len(argv):
			a = argv[i]
			if a == '--child':
				i += 1
				continue
			if a == '--adam-on':
				child_adam = argv[i+1] == '1'
				i += 2
				continue
			if a == '--hess-on':
				child_hess = argv[i+1] == '1'
				i += 2
				continue
			if a == '--tag':
				child_tag = argv[i+1]
				i += 2
				continue
			if a == '--epochs':
				child_epochs = int(argv[i+1]); i += 2; continue
			if a == '--batch-size':
				child_batch = int(argv[i+1]); i += 2; continue
			if a == '--lr-adam':
				child_lr_adam = float(argv[i+1]); i += 2; continue
			if a == '--lr-hessian':
				child_lr_hessian = float(argv[i+1]); i += 2; continue
			if a == '--hessian-damping':
				child_hessian_damping = float(argv[i+1]); i += 2; continue
			if a == '--hutchinson-samples':
				child_hutchinson_samples = int(argv[i+1]); i += 2; continue
			if a == '--switch-every':
				child_switch_every = int(argv[i+1]); i += 2; continue
			if a == '--start-with-adam':
				child_start_with_adam = bool(int(argv[i+1])); i += 2; continue
			if a == '--subset':
				child_subset = int(argv[i+1]); i += 2; continue
			if a == '--seed':
				child_seed = int(argv[i+1]); i += 2; continue
			if a == '--log-interval':
				child_log_interval = int(argv[i+1]); i += 2; continue
			i += 1

	a = SimpleNamespace()
	a.epochs = child_epochs
	a.batch_size = child_batch
	a.lr_adam = child_lr_adam
	a.lr_hessian = child_lr_hessian
	a.hessian_damping = child_hessian_damping
	a.hutchinson_samples = child_hutchinson_samples
	a.switch_every = child_switch_every
	a.start_with_adam = child_start_with_adam
	a.subset = child_subset
	a.seed = child_seed
	a.cpu = FORCE_CPU
	a.log_interval = child_log_interval
	# child overrides
	a.adam_on = child_adam
	a.hess_on = child_hess
	a.tag = child_tag

	train(a)

	if LAUNCH_PARALLEL or is_launch:
		launch_and_plot()



