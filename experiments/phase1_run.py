import sys, os, json, time

# Auto-detect project root regardless of where script is run from
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
_RESULTS = os.path.join(_HERE, "results")
sys.path.insert(0, _ROOT)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from pragma.benchmark.generator import generate_benchmark
from pragma.benchmark.evaluator import evaluate_baseline, brier_score
from pragma.baselines.neural import NeuralBaseline
from pragma.baselines.symbolic import SymbolicBaseline
from pragma.baselines.bayesian import BayesianBaseline
from pragma.baselines.world_model import WorldModelBaseline

ARC_COLORS = ['#000000','#0074D9','#FF4136','#2ECC40','#FFDC00',
              '#AAAAAA','#F012BE','#FF851B','#7FDBFF','#870C25']

def run_phase1():
    print("=" * 70)
    print("PRAGMA PHASE 1: BASELINE EXPERIMENT")
    print("=" * 70)
    print()

    print("Generating benchmark (10 tasks x 10 types = 100 tasks)...")
    tasks = generate_benchmark(n_tasks_per_type=10, seed=42)
    print(f"  Generated {len(tasks)} tasks")
    task_types = list(set(t.name for t in tasks))
    diff_counts = {}
    for t in tasks:
        diff_counts[t.difficulty] = diff_counts.get(t.difficulty, 0) + 1
    print(f"  Task types: {task_types}")
    print(f"  Difficulty: {diff_counts}")
    print()

    neural   = NeuralBaseline(k=3)
    symbolic = SymbolicBaseline(confidence_threshold=0.85)
    bayesian = BayesianBaseline(temperature=2.0)
    wm       = WorldModelBaseline(latent_dim=32)

    baselines = [("Neural", neural), ("Symbolic", symbolic),
                 ("Bayesian", bayesian), ("World Model", wm)]
    all_results, all_predictions = {}, {}

    for name, baseline in baselines:
        print(f"Running {name} baseline...")
        t0 = time.time()
        preds = baseline.solve_batch(tasks)
        elapsed = time.time() - t0
        all_predictions[name] = preds

        uncertainties = None
        if name == "Bayesian":
            unc_data = bayesian.solve_batch_with_uncertainty(tasks)
            uncertainties = []
            for (pred, unc_info), task in zip(unc_data, tasks):
                gt_flat = task.test_output.flatten()
                cell_probs = unc_info.get("cell_probs", None)
                if cell_probs is not None and pred is not None:
                    n_cells = cell_probs.shape[0]
                    gt_trunc = gt_flat[:n_cells] if len(gt_flat) >= n_cells else np.pad(gt_flat, (0, n_cells - len(gt_flat)))
                    uncertainties.append((cell_probs, gt_trunc))
            uncertainties = [u for u in uncertainties if u is not None]

        results = evaluate_baseline(tasks, preds, uncertainties=uncertainties if name == "Bayesian" else None)
        results["time_seconds"] = elapsed
        results["time_per_task"] = elapsed / len(tasks)
        all_results[name] = results

        print(f"  Done in {elapsed:.2f}s")
        print(f"  Exact match:    {results['exact_match']:.3f}")
        print(f"  Pixel accuracy: {results['pixel_accuracy']:.3f}")
        print(f"  Shape match:    {results['shape_match']:.3f}")
        if results.get("brier_score"):
            print(f"  Brier score:    {results['brier_score']:.4f}")
        print()

    os.makedirs(_RESULTS, exist_ok=True)
    with open(os.path.join(_RESULTS, 'phase1_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to: {_RESULTS}")
    print()

    _plot_comparison(all_results)
    _plot_by_type(all_results, task_types)
    _plot_by_diff(all_results)
    _plot_failures(tasks, all_predictions)
    _plot_uncertainty(tasks, bayesian)
    print("All figures saved.")
    print()

    _summary(all_results)
    return all_results, tasks, all_predictions


def _plot_comparison(all_results):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("PRAGMA Phase 1: Four-Baseline Comparison (100 tasks)", fontsize=13, fontweight='bold')
    names = list(all_results.keys())
    colors = ['#2196F3','#4CAF50','#FF9800','#9C27B0']
    for ax, metric, label in zip(axes,
        ['exact_match','pixel_accuracy','shape_match'],
        ['Exact Match\n(Primary)','Pixel Accuracy','Shape Match']):
        vals = [all_results[n][metric] for n in names]
        bars = ax.bar(names, vals, color=colors, alpha=0.85, edgecolor='white', linewidth=1.5)
        ax.set_ylim(0, 1.1); ax.set_ylabel(label); ax.set_title(label, fontweight='bold')
        ax.tick_params(axis='x', rotation=15); ax.spines[['top','right']].set_visible(False)
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.4)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(_RESULTS, 'fig1_main_comparison.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def _plot_by_type(all_results, task_types):
    names = list(all_results.keys())
    colors = ['#2196F3','#4CAF50','#FF9800','#9C27B0']
    fig, ax = plt.subplots(figsize=(16, 6))
    x = np.arange(len(task_types)); width = 0.2
    for i, (name, color) in enumerate(zip(names, colors)):
        vals = [all_results[name]['by_task_type'].get(tt, 0.0) for tt in task_types]
        ax.bar(x + i*width, vals, width, label=name, color=color, alpha=0.85, edgecolor='white')
    ax.set_xticks(x + 1.5*width)
    ax.set_xticklabels([t.replace('_','\n') for t in task_types], fontsize=8)
    ax.set_ylim(0,1.15); ax.set_ylabel("Exact Match"); ax.legend()
    ax.set_title("Performance by Task Type", fontweight='bold')
    ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(_RESULTS, 'fig2_by_task_type.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def _plot_by_diff(all_results):
    names = list(all_results.keys())
    colors = ['#2196F3','#4CAF50','#FF9800','#9C27B0']
    fig, ax = plt.subplots(figsize=(10, 5))
    diffs = ['diff_1','diff_2','diff_3']; labels = ['Easy','Medium','Hard']
    x = np.arange(3); width = 0.2
    for i, (name, color) in enumerate(zip(names, colors)):
        vals = [all_results[name]['by_difficulty'].get(d, 0.0) for d in diffs]
        ax.bar(x + i*width, vals, width, label=name, color=color, alpha=0.85, edgecolor='white')
    ax.set_xticks(x + 1.5*width); ax.set_xticklabels(labels)
    ax.set_ylim(0,1.15); ax.set_ylabel("Exact Match"); ax.legend()
    ax.set_title("Performance by Difficulty", fontweight='bold')
    ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(_RESULTS, 'fig3_by_difficulty.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def _plot_failures(tasks, all_predictions):
    names = list(all_predictions.keys())
    task_types = sorted(set(t.name for t in tasks))
    matrix = []
    for name in names:
        row = []
        for tt in task_types:
            matching = [(t,p) for t,p in zip(tasks, all_predictions[name]) if t.name == tt]
            fails = sum(1 for t,p in matching if p is None or not np.array_equal(p, t.test_output))
            row.append(fails / max(len(matching), 1))
        matrix.append(row)
    matrix = np.array(matrix)
    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(matrix, cmap='RdYlGn_r', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(len(task_types)))
    ax.set_xticklabels([t.replace('_','\n') for t in task_types], fontsize=8)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    ax.set_title("Failure Rate Heatmap (red=fails, green=succeeds)", fontweight='bold')
    plt.colorbar(im, ax=ax, label="Failure Rate")
    for i in range(len(names)):
        for j in range(len(task_types)):
            ax.text(j, i, f"{matrix[i,j]:.1f}", ha='center', va='center', fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(_RESULTS, 'fig4_failure_heatmap.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def _plot_uncertainty(tasks, bayesian):
    unc_results = bayesian.solve_batch_with_uncertainty(tasks)
    confidences, accuracies, entropies = [], [], []
    for (pred, unc), task in zip(unc_results, tasks):
        if pred is None: continue
        confidences.append(unc.get("map_confidence", 0.5))
        accuracies.append(float(np.array_equal(pred, task.test_output)))
        entropies.append(unc.get("entropy", 0.0))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Bayesian Baseline: Uncertainty Analysis", fontsize=13, fontweight='bold')
    bins = np.linspace(0,1,11)
    bcs, bas = [], []
    for i in range(len(bins)-1):
        mask = [bins[i] <= c < bins[i+1] for c in confidences]
        if any(mask):
            bcs.append(np.mean([c for c,m in zip(confidences,mask) if m]))
            bas.append(np.mean([a for a,m in zip(accuracies,mask) if m]))
    axes[0].plot([0,1],[0,1],'k--',alpha=0.5,label='Perfect')
    axes[0].scatter(bcs, bas, s=100, zorder=3, label='Bayesian')
    axes[0].set_xlabel("Confidence"); axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Calibration Curve"); axes[0].legend()
    axes[0].spines[['top','right']].set_visible(False)
    axes[1].hist(entropies, bins=20, color='#FF9800', edgecolor='white', alpha=0.85)
    axes[1].axvline(np.mean(entropies), color='red', linestyle='--', label=f'Mean={np.mean(entropies):.2f}')
    axes[1].set_xlabel("Posterior Entropy"); axes[1].set_ylabel("Count")
    axes[1].set_title("Uncertainty Distribution"); axes[1].legend()
    axes[1].spines[['top','right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(_RESULTS, 'fig5_uncertainty.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def _summary(all_results):
    print("=" * 70)
    print("PHASE 1 RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Baseline':<14} {'Exact Match':>12} {'Pixel Acc':>10} {'Time/task':>10}")
    print("-" * 50)
    for name, res in all_results.items():
        print(f"{name:<14} {res['exact_match']:>12.3f} {res['pixel_accuracy']:>10.3f} {res.get('time_per_task',0):>9.4f}s")
    print("-" * 50)
    print()
    print(f"Results and figures saved to: {_RESULTS}")
    print("=" * 70)


if __name__ == "__main__":
    run_phase1()
