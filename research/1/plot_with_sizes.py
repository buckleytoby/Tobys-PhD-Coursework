# Quick plotter that annotates legend with model sizes
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import importlib.util

HERE = os.path.dirname(__file__)
METRICS = os.path.join(HERE, 'metrics')
# files to compare (explicit runs)
HILL_FILE = os.path.join(METRICS, 'metrics_hill_explicit.npz')
TRANS_FILE = os.path.join(METRICS, 'metrics_transformer_explicit.npz')
# MLP runs (added later)
MLP_SHALLOW_FILE = os.path.join(METRICS, 'metrics_mlp_shallow.npz')
MLP_DEEP_FILE = os.path.join(METRICS, 'metrics_mlp_deep.npz')
CNN_FILE = os.path.join(METRICS, 'metrics_cnn.npz')
OUT = os.path.join(METRICS, 'comparison_all_with_sizes.png')

# load the models from the main script to compute sizes
spec = importlib.util.spec_from_file_location('job', os.path.join(HERE, '1.py'))
job = importlib.util.module_from_spec(spec)
spec.loader.exec_module(job)

# instantiate models with the same config the script currently uses
hill_model = job.HillClassifier(in_dim=28*28, inner_dim=job.HILL_INNER_DIM, stages=job.HILL_STAGES)
transformer_model = job.TransformerClassifier(img_size=28, patch_size=7, emb_dim=276, n_heads=4, n_layers=4, mlp_dim=1104)
# instantiate the MLP variants if available in the main script
shallow_mlp = None
deep_mlp = None
if hasattr(job, 'ShallowWideMLP'):
	shallow_mlp = job.ShallowWideMLP()
if hasattr(job, 'NarrowDeepMLP'):
	deep_mlp = job.NarrowDeepMLP()
cnn_model = None
if hasattr(job, 'SimpleCNN'):
	cnn_model = job.SimpleCNN()

hill_params = sum(p.numel() for p in hill_model.parameters())
trans_params = sum(p.numel() for p in transformer_model.parameters())
shallow_params = sum(p.numel() for p in shallow_mlp.parameters()) if shallow_mlp is not None else None
deep_params = sum(p.numel() for p in deep_mlp.parameters()) if deep_mlp is not None else None
cnn_params = sum(p.numel() for p in cnn_model.parameters()) if cnn_model is not None else None

# load data if files exist
def load_if_exists(path):
	if os.path.exists(path):
		return np.load(path)
	return None

D_h = load_if_exists(HILL_FILE)
D_t = load_if_exists(TRANS_FILE)
D_ms = load_if_exists(MLP_SHALLOW_FILE)
D_md = load_if_exists(MLP_DEEP_FILE)
D_cnn = load_if_exists(CNN_FILE)

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
if D_h is not None:
	plt.plot(D_h['steps'], D_h['loss'], label=f'Hill ({hill_params:,} params)', color='tab:orange')
if D_t is not None:
	plt.plot(D_t['steps'], D_t['loss'], label=f'Transformer ({trans_params:,} params)', color='tab:blue')
if D_ms is not None and shallow_params is not None:
	plt.plot(D_ms['steps'], D_ms['loss'], label=f'MLP shallow ({shallow_params:,} params)', color='tab:green')
if D_md is not None and deep_params is not None:
	plt.plot(D_md['steps'], D_md['loss'], label=f'MLP deep ({deep_params:,} params)', color='tab:purple')
if D_cnn is not None and cnn_params is not None:
	plt.plot(D_cnn['steps'], D_cnn['loss'], label=f'CNN ({cnn_params:,} params)', color='tab:brown')
plt.xlabel('step')
plt.ylabel('loss')
plt.legend()
plt.title('Loss')

plt.subplot(1,2,2)
if D_h is not None:
	plt.plot(D_h['steps'], D_h['acc'], label=f'Hill ({hill_params:,} params)', color='tab:orange')
if D_t is not None:
	plt.plot(D_t['steps'], D_t['acc'], label=f'Transformer ({trans_params:,} params)', color='tab:blue')
if D_ms is not None and shallow_params is not None:
	plt.plot(D_ms['steps'], D_ms['acc'], label=f'MLP shallow ({shallow_params:,} params)', color='tab:green')
if D_md is not None and deep_params is not None:
	plt.plot(D_md['steps'], D_md['acc'], label=f'MLP deep ({deep_params:,} params)', color='tab:purple')
if D_cnn is not None and cnn_params is not None:
	plt.plot(D_cnn['steps'], D_cnn['acc'], label=f'CNN ({cnn_params:,} params)', color='tab:brown')
plt.xlabel('step')
plt.ylabel('accuracy (%)')
plt.legend()
plt.title('Accuracy')

plt.tight_layout()
plt.savefig(OUT)
print('Saved', OUT)
