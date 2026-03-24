import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

"""
PRAGMA Phase 2 — Experiments 2.3 & 2.4: Full Binary Integration Suite
=======================================================================
Exp 2.3: Neural + Bayesian (uncertainty-guided neural prediction)
Exp 2.4: All-four combo vote (preview of Phase 3 integration)

HOW TO RUN:
  python3 phase2_exp3_4.py
"""

import sys, os, json, time
import os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pragma.benchmark.generator import generate_benchmark
from pragma.benchmark.evaluator import evaluate_baseline, brier_score
from pragma.baselines.neural import NeuralBaseline, extract_features, apply_learned_transformation
from pragma.baselines.symbolic import SymbolicBaseline
from pragma.baselines.bayesian import BayesianBaseline
from pragma.baselines.world_model import WorldModelBaseline


# ─── Exp 2.3: Neural + Bayesian ───────────────────────────────────────────────

class NeuralBayesianBaseline:
    """
    Neural prediction FILTERED by Bayesian uncertainty.
    Key idea: use Bayesian posterior entropy to decide whether to trust
    the neural prediction or abstain/fall back to safest guess.

    When Bayesian entropy is LOW  (confident about rule) → use Bayesian/Symbolic prediction
    When Bayesian entropy is HIGH (uncertain)            → use Neural generalization
    This inverts the NeuSym logic — neural is the backup, Bayesian is the gatekeeper.
    """
    def __init__(self, entropy_threshold=1.5):
        self.entropy_threshold = entropy_threshold
        self._bayes  = BayesianBaseline(temperature=2.0)
        self._neural = NeuralBaseline(k=3)

    def solve(self, task):
        bay_pred, unc = self._bayes.solve_with_uncertainty(task)
        entropy = unc['entropy']

        if entropy <= self.entropy_threshold:
            # Bayesian is confident → use its MAP prediction
            return bay_pred, 'bayesian_confident', entropy
        else:
            # Bayesian is uncertain → neural generalization
            neu_pred = self._neural.solve(task)
            return neu_pred, 'neural_fallback', entropy

    def solve_batch(self, tasks):
        preds, routing, entropies = [], {}, []
        for task in tasks:
            pred, route, ent = self.solve(task)
            preds.append(pred)
            routing[route] = routing.get(route, 0) + 1
            entropies.append(ent)
        return preds, routing, entropies


# ─── Exp 2.4: All-Four Ensemble Vote ──────────────────────────────────────────

class EnsembleBaseline:
    """
    Preview of Phase 3: all four paradigms vote.
    Each paradigm predicts independently. Majority pixel-wise vote.
    Confidence-weighted: symbolic/bayesian votes count more when confident.
    """
    def __init__(self):
        self._neural   = NeuralBaseline(k=3)
        self._symbolic = SymbolicBaseline()
        self._bayesian = BayesianBaseline()
        self._wm       = WorldModelBaseline()

    def solve(self, task):
        preds = []
        weights = []

        neu_pred = self._neural.solve(task)
        if neu_pred is not None and neu_pred.shape == task.test_input.shape:
            preds.append(neu_pred); weights.append(1.0)

        sym_pred, expl = self._symbolic.solve_with_explanation(task)
        if sym_pred is not None:
            preds.append(sym_pred); weights.append(1.0 + expl['confidence'])

        bay_pred, unc = self._bayesian.solve_with_uncertainty(task)
        if bay_pred is not None:
            preds.append(bay_pred); weights.append(1.0 + unc['map_confidence'])

        wm_pred = self._wm.solve(task)
        if wm_pred is not None and wm_pred.shape == task.test_output.shape:
            preds.append(wm_pred); weights.append(0.8)

        if not preds: return None

        # Find the most common output shape
        from collections import Counter
        shape_counts = Counter(p.shape for p in preds)
        target_shape = shape_counts.most_common(1)[0][0]
        preds_filtered = [(p, w) for p, w in zip(preds, weights) if p.shape == target_shape]
        if not preds_filtered: return preds[0]

        # Weighted pixel-wise vote
        h, w = target_shape
        vote_grid = np.zeros((h, w, 10), dtype=float)
        total_w = sum(wt for _, wt in preds_filtered)
        for pred, wt in preds_filtered:
            for r in range(h):
                for c in range(w):
                    vote_grid[r, c, int(pred[r, c])] += wt / total_w

        return np.argmax(vote_grid, axis=2).astype(np.int32)

    def solve_batch(self, tasks):
        return [self.solve(t) for t in tasks]


# ─── Main Runner ──────────────────────────────────────────────────────────────

def run_experiments():
    print("=" * 70)
    print("PRAGMA PHASE 2 — EXPERIMENTS 2.3 & 2.4")
    print("=" * 70)
    print()

    tasks = generate_benchmark(n_tasks_per_type=10, seed=42)
    task_types = sorted(set(t.name for t in tasks))

    # Phase 1 references
    sym_res = evaluate_baseline(tasks, SymbolicBaseline().solve_batch(tasks))
    neu_res = evaluate_baseline(tasks, NeuralBaseline().solve_batch(tasks))

    # ── Exp 2.3: Neural + Bayesian ────────────────────────────────────────
    print("▶ EXP 2.3: Neural + Bayesian (uncertainty-gated)")
    neubayes = NeuralBayesianBaseline(entropy_threshold=1.5)
    t0 = time.time()
    nb_preds, nb_routing, nb_entropies = neubayes.solve_batch(tasks)
    nb_res = evaluate_baseline(tasks, nb_preds)
    nb_res['routing'] = nb_routing
    print(f"  EM={nb_res['exact_match']:.3f}  Routing={nb_routing}")
    print(f"  Mean entropy: {np.mean(nb_entropies):.3f}")
    print()

    # ── Exp 2.4: All-Four Ensemble ────────────────────────────────────────
    print("▶ EXP 2.4: All-Four Ensemble Vote (Phase 3 preview)")
    ensemble = EnsembleBaseline()
    t0 = time.time()
    ens_preds = ensemble.solve_batch(tasks)
    elapsed = time.time() - t0
    ens_res = evaluate_baseline(tasks, ens_preds)
    print(f"  EM={ens_res['exact_match']:.3f}  time={elapsed:.2f}s")
    print()

    # ── Grand Summary ─────────────────────────────────────────────────────
    print("=" * 70)
    print("PHASE 2 GRAND RESULTS (all binary integrations)")
    print("=" * 70)

    all_methods = {
        'Neural (P1)':         neu_res['exact_match'],
        'Symbolic (P1)':       sym_res['exact_match'],
        'Neural+Bayesian':     nb_res['exact_match'],
        'All-Four Ensemble':   ens_res['exact_match'],
    }

    for method, em in sorted(all_methods.items(), key=lambda x: -x[1]):
        bar = '█' * int(em * 30)
        print(f"  {method:<24} {em:.3f}  {bar}")

    print()
    best_phase1 = max(neu_res['exact_match'], sym_res['exact_match'])
    best_phase2 = max(nb_res['exact_match'], ens_res['exact_match'])
    print(f"  Best Phase 1 single:  {best_phase1:.3f}")
    print(f"  Best Phase 2 hybrid:  {best_phase2:.3f}")
    print(f"  Improvement:          {best_phase2 - best_phase1:+.3f}")
    print()

    # Per task type for ensemble
    print(f"  {'Task Type':<22} {'Symbolic':>9} {'Ensemble':>9} {'Delta':>7}")
    print("  " + "─" * 55)
    for tt in task_types:
        s  = sym_res['by_task_type'].get(tt, 0.0)
        e  = ens_res['by_task_type'].get(tt, 0.0)
        print(f"  {tt:<22} {s:>9.2f} {e:>9.2f} {e-s:>+6.2f}")

    print()
    print("SEND THIS ENTIRE OUTPUT BACK.")
    print("=" * 70)

    # Save
    os.makedirs(RESULTS_DIR + '/', exist_ok=True)
    with open(RESULTS_DIR + '//phase2_exp3_4.json', 'w') as f:
        json.dump({'neural_bayesian': nb_res, 'ensemble': ens_res}, f, indent=2, default=str)

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Phase 2 Experiments 2.3 & 2.4", fontsize=13, fontweight='bold')

    # Left: bar chart all methods
    ax = axes[0]
    methods = list(all_methods.keys())
    vals = [all_methods[m] for m in methods]
    colors = ['#2196F3','#4CAF50','#FF9800','#E91E63']
    bars = ax.bar(methods, vals, color=colors, alpha=0.85, edgecolor='white')
    ax.set_ylim(0, 1.1)
    ax.axhline(best_phase1, color='red', linestyle='--', alpha=0.5, label='Best Phase 1')
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, val+0.01, f'{val:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.legend(); ax.set_ylabel("Exact Match")
    ax.set_title("All Methods: Overall Accuracy", fontweight='bold')
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(axis='x', rotation=15)

    # Right: per-task-type heatmap
    ax = axes[1]
    methods_tt = ['Neural', 'Symbolic', 'N+Bayes', 'Ensemble']
    matrix = np.array([
        [neu_res['by_task_type'].get(tt,0) for tt in task_types],
        [sym_res['by_task_type'].get(tt,0) for tt in task_types],
        [nb_res['by_task_type'].get(tt,0) for tt in task_types],
        [ens_res['by_task_type'].get(tt,0) for tt in task_types],
    ])
    im = ax.imshow(matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(len(task_types)))
    ax.set_xticklabels([t.replace('_','\n') for t in task_types], fontsize=7)
    ax.set_yticks(range(len(methods_tt)))
    ax.set_yticklabels(methods_tt)
    ax.set_title("Per-Task-Type Accuracy\n(green=100%, red=0%)", fontweight='bold')
    for i in range(len(methods_tt)):
        for j in range(len(task_types)):
            ax.text(j, i, f"{matrix[i,j]:.1f}", ha='center', va='center', fontsize=7)
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR + '//phase2_exp3_4.png',
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("\n  Figures saved.")

if __name__ == "__main__":
    run_experiments()
