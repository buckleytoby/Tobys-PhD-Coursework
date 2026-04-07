"""
Load metrics .npz files and plot comparisons for loss and accuracy.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

METRICS_DIR = os.path.join(os.path.dirname(__file__), 'metrics')

CASES = [
    ('adam_off', 'ADAM_ON=0,HESS_ON=1'),
    ('hess_off', 'ADAM_ON=1,HESS_ON=0'),
    ('both_on', 'ADAM_ON=1,HESS_ON=1'),
]

def load_metrics(tag):
    p = os.path.join(METRICS_DIR, f'metrics_{tag}.npz')
    if not os.path.exists(p):
        return None
    d = np.load(p)
    return d


def main():
    rows = []
    plt.figure(figsize=(10,4))
    # loss subplot
    plt.subplot(1,2,1)
    for tag, label in CASES:
        d = load_metrics(tag)
        if d is None:
            print('Missing', tag)
            continue
        steps = d['steps']
        loss = d['loss']
        # smooth
        plt.plot(steps, loss, label=label)
    plt.xlabel('step')
    plt.ylabel('loss')
    plt.legend()
    plt.title('Loss over time')

    # acc subplot
    plt.subplot(1,2,2)
    for tag, label in CASES:
        d = load_metrics(tag)
        if d is None:
            continue
        steps = d['steps']
        acc = d['acc']
        plt.plot(steps, acc, label=label)
    plt.xlabel('step')
    plt.ylabel('accuracy (%)')
    plt.legend()
    plt.title('Accuracy per-batch')

    out = os.path.join(METRICS_DIR, 'comparison.png')
    plt.tight_layout()
    plt.savefig(out)
    print('Saved comparison plot to', out)

if __name__ == '__main__':
    main()
