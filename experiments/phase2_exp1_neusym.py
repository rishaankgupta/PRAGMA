import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

"""
PRAGMA Phase 2 — Experiment 2.1: NeuSym (Neural + Symbolic)
=============================================================
HYPOTHESIS: Combining neural feature matching WITH symbolic rule
induction outperforms either alone — specifically on tasks where
one paradigm fails and the other succeeds.

ARCHITECTURE:
  Stage 1 (Neural): extract features, compute similarity scores
  Stage 2 (Symbolic): run all rule detectors, get confidences
  Stage 3 (Fusion): weighted combination of both signals
  Stage 4: pick best prediction using fused score

PREDICTION: NeuSym should close the gap on:
  - count_encode   (Symbolic=0%,  Neural=100%) → NeuSym should stay ~100%
  - object_move    (Symbolic=20%, Neural=10%)  → NeuSym should improve
  - rotate_90cw    (Symbolic=100%, Neural=0%)  → NeuSym should stay ~100%

HOW TO RUN:
  python3 phase2_neusym.py
  (from the /home/claude directory or wherever pragma/ lives)

SEND ME: the full terminal output + the numbers in the RESULTS table
"""

import sys, os, json, time
import os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pragma.benchmark.generator import generate_benchmark
from pragma.benchmark.evaluator import evaluate_baseline, exact_match, pixel_accuracy
from pragma.baselines.neural import NeuralBaseline, extract_features, apply_learned_transformation
from pragma.baselines.symbolic import SymbolicBaseline, RULES


# ─── NeuSym: The Hybrid ───────────────────────────────────────────────────────

class NeuSymBaseline:
    """
    Neural + Symbolic fusion.

    For each test input:
      1. Neural arm: find most similar training input by feature cosine similarity
         → produces: candidate prediction + similarity score
      2. Symbolic arm: run all rule detectors on training pairs
         → produces: best rule + confidence score
      3. Fusion: if symbolic confidence > threshold, use symbolic answer
                 else if neural similarity > threshold, use neural answer
                 else run voting between all candidate predictions

    Key innovation: the two arms are complementary. When symbolic is
    certain (high confidence), trust it. When it's uncertain (low confidence),
    fall back to neural generalization.
    """

    def __init__(self,
                 symbolic_threshold=0.85,
                 neural_threshold=0.75,
                 fusion_mode='confidence_gated'):
        self.symbolic_threshold = symbolic_threshold
        self.neural_threshold   = neural_threshold
        self.fusion_mode        = fusion_mode
        self._symbolic = SymbolicBaseline(confidence_threshold=symbolic_threshold)
        self._neural   = NeuralBaseline(k=3)

    def _get_symbolic_prediction(self, task):
        pred, explanation = self._symbolic.solve_with_explanation(task)
        return pred, explanation['confidence'], explanation['selected_rule']

    def _get_neural_prediction(self, task):
        train = task.train_pairs
        test_inp = task.test_input
        if not train: return None, 0.0

        train_feats = np.array([extract_features(inp) for inp, _ in train])
        test_feat   = extract_features(test_inp)

        sims = []
        for tf in train_feats:
            sim = np.dot(test_feat, tf) / (np.linalg.norm(test_feat) * np.linalg.norm(tf) + 1e-9)
            sims.append(float(sim))

        best_idx = int(np.argmax(sims))
        best_sim = sims[best_idx]
        ref_inp, ref_out = train[best_idx]
        pred = apply_learned_transformation(test_inp, ref_inp, ref_out)
        return pred, best_sim

    def solve(self, task):
        sym_pred, sym_conf, sym_rule = self._get_symbolic_prediction(task)
        neu_pred, neu_sim            = self._get_neural_prediction(task)

        if self.fusion_mode == 'confidence_gated':
            # Trust symbolic if it's confident
            if sym_conf >= self.symbolic_threshold and sym_pred is not None:
                return sym_pred, 'symbolic', sym_conf, sym_rule
            # Fall back to neural if sufficiently similar
            if neu_sim >= self.neural_threshold and neu_pred is not None:
                return neu_pred, 'neural', neu_sim, 'feature_match'
            # Both uncertain: prefer symbolic (more principled)
            if sym_pred is not None:
                return sym_pred, 'symbolic_uncertain', sym_conf, sym_rule
            return neu_pred, 'neural_fallback', neu_sim, 'feature_match'

        elif self.fusion_mode == 'max_confidence':
            # Just pick whichever arm is more confident
            if sym_conf >= neu_sim:
                return sym_pred, 'symbolic', sym_conf, sym_rule
            else:
                return neu_pred, 'neural', neu_sim, 'feature_match'

        elif self.fusion_mode == 'vote':
            # Majority vote over all candidate predictions
            candidates = []
            if sym_pred is not None: candidates.append(sym_pred)
            if neu_pred is not None: candidates.append(neu_pred)
            if not candidates: return None, 'none', 0.0, 'none'
            # Pick the candidate with highest agreement
            best_cand = candidates[0]
            best_agreement = 0
            for cand in candidates:
                agreement = sum(
                    np.array_equal(cand, other) for other in candidates
                )
                if agreement > best_agreement:
                    best_agreement = agreement
                    best_cand = cand
            return best_cand, 'vote', best_agreement / len(candidates), 'vote'

    def solve_batch(self, tasks):
        results = []
        decisions = {'symbolic': 0, 'neural': 0, 'symbolic_uncertain': 0,
                     'neural_fallback': 0, 'vote': 0, 'none': 0}
        for task in tasks:
            pred, decision, conf, rule = self.solve(task)
            results.append(pred)
            decisions[decision] = decisions.get(decision, 0) + 1
        return results, decisions


# ─── Experiment Runner ────────────────────────────────────────────────────────

def run_experiment():
    print("=" * 70)
    print("PRAGMA PHASE 2 — EXPERIMENT 2.1: NeuSym (Neural + Symbolic)")
    print("=" * 70)
    print()

    # Load same benchmark (same seed = same tasks as Phase 1)
    print("▶ Loading benchmark...")
    tasks = generate_benchmark(n_tasks_per_type=10, seed=42)
    task_types = sorted(set(t.name for t in tasks))
    print(f"  ✓ {len(tasks)} tasks loaded")
    print()

    # Phase 1 baselines for comparison
    neural_baseline   = NeuralBaseline(k=3)
    symbolic_baseline = SymbolicBaseline(confidence_threshold=0.85)

    print("▶ Running Phase 1 baselines (reference)...")
    neural_preds   = neural_baseline.solve_batch(tasks)
    symbolic_preds = symbolic_baseline.solve_batch(tasks)
    neural_results   = evaluate_baseline(tasks, neural_preds)
    symbolic_results = evaluate_baseline(tasks, symbolic_preds)
    print(f"  Neural:   EM={neural_results['exact_match']:.3f}")
    print(f"  Symbolic: EM={symbolic_results['exact_match']:.3f}")
    print()

    # NeuSym in three fusion modes
    fusion_modes = ['confidence_gated', 'max_confidence', 'vote']
    neusym_results_all = {}

    for mode in fusion_modes:
        print(f"▶ Running NeuSym [{mode}]...")
        neusym = NeuSymBaseline(fusion_mode=mode)
        t0 = time.time()
        preds, decisions = neusym.solve_batch(tasks)
        elapsed = time.time() - t0
        results = evaluate_baseline(tasks, preds)
        results['decisions'] = decisions
        results['time'] = elapsed
        neusym_results_all[mode] = results
        print(f"  ✓ EM={results['exact_match']:.3f}  PA={results['pixel_accuracy']:.3f}  time={elapsed:.3f}s")
        print(f"    Decision breakdown: {decisions}")
        print()

    # Best NeuSym
    best_mode = max(fusion_modes, key=lambda m: neusym_results_all[m]['exact_match'])
    best_results = neusym_results_all[best_mode]

    # ── Per-Task-Type Breakdown ───────────────────────────────────────────
    print("▶ Per-task-type breakdown (CRITICAL — identifies synergy):")
    print(f"  {'Task Type':<22} {'Neural':>8} {'Symbolic':>9} {'NeuSym':>8} {'Delta':>8}")
    print("  " + "─" * 60)

    synergy_tasks = []
    regression_tasks = []

    for tt in task_types:
        n_em  = neural_results['by_task_type'].get(tt, 0.0)
        s_em  = symbolic_results['by_task_type'].get(tt, 0.0)
        ns_em = best_results['by_task_type'].get(tt, 0.0)
        best_prev = max(n_em, s_em)
        delta = ns_em - best_prev
        flag = ""
        if delta > 0.05:
            flag = "✓ SYNERGY"
            synergy_tasks.append(tt)
        elif delta < -0.05:
            flag = "✗ REGRESSION"
            regression_tasks.append(tt)
        print(f"  {tt:<22} {n_em:>8.2f} {s_em:>9.2f} {ns_em:>8.2f} {delta:>+7.2f}  {flag}")

    print()
    print(f"  Synergy tasks    (NeuSym > best single): {synergy_tasks}")
    print(f"  Regression tasks (NeuSym < best single): {regression_tasks}")
    print()

    # ── Summary Table ─────────────────────────────────────────────────────
    print("=" * 70)
    print("EXPERIMENT 2.1 RESULTS")
    print("=" * 70)
    print(f"  {'Method':<26} {'Exact Match':>12} {'Pixel Acc':>10}")
    print("  " + "─" * 52)
    print(f"  {'Neural (Phase 1)':<26} {neural_results['exact_match']:>12.3f} {neural_results['pixel_accuracy']:>10.3f}")
    print(f"  {'Symbolic (Phase 1)':<26} {symbolic_results['exact_match']:>12.3f} {symbolic_results['pixel_accuracy']:>10.3f}")
    for mode in fusion_modes:
        r = neusym_results_all[mode]
        label = f"NeuSym [{mode}]"
        print(f"  {label:<26} {r['exact_match']:>12.3f} {r['pixel_accuracy']:>10.3f}")
    print("  " + "─" * 52)
    best_single = max(neural_results['exact_match'], symbolic_results['exact_match'])
    best_neusym = max(neusym_results_all[m]['exact_match'] for m in fusion_modes)
    improvement = best_neusym - best_single
    print(f"  Improvement over best single: {improvement:+.3f}")
    print()

    if improvement > 0:
        print("  ✓ HYPOTHESIS CONFIRMED: NeuSym > both individual paradigms")
        print("    Integration produces synergy, not just combination.")
    elif improvement == 0:
        print("  → NEUTRAL: NeuSym matches best single paradigm.")
        print("    Integration does not regress — safe baseline for Phase 3.")
    else:
        print("  ✗ HYPOTHESIS REJECTED: NeuSym < best single paradigm.")
        print("    Fusion is hurting — need to redesign the interface.")

    print()
    print("=" * 70)
    print("SEND THIS ENTIRE OUTPUT TO YOUR RESEARCH PARTNER")
    print("They will analyze and design Experiment 2.2 based on these results.")
    print("=" * 70)

    # ── Save Results ──────────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR + '/', exist_ok=True)
    save_data = {
        'neural': neural_results,
        'symbolic': symbolic_results,
        'neusym': neusym_results_all,
        'best_mode': best_mode,
        'improvement': improvement,
        'synergy_tasks': synergy_tasks,
        'regression_tasks': regression_tasks
    }
    with open(RESULTS_DIR + '//phase2_exp1_neusym.json', 'w') as f:
        json.dump(save_data, f, indent=2, default=str)

    # ── Figure ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Phase 2 Exp 2.1: NeuSym vs Individual Paradigms", fontsize=13, fontweight='bold')

    # Left: per-task-type comparison
    ax = axes[0]
    x = np.arange(len(task_types))
    w = 0.25
    n_vals  = [neural_results['by_task_type'].get(tt,0) for tt in task_types]
    s_vals  = [symbolic_results['by_task_type'].get(tt,0) for tt in task_types]
    ns_vals = [best_results['by_task_type'].get(tt,0) for tt in task_types]
    ax.bar(x - w, n_vals,  w, label='Neural',   color='#2196F3', alpha=0.85)
    ax.bar(x,     s_vals,  w, label='Symbolic',  color='#4CAF50', alpha=0.85)
    ax.bar(x + w, ns_vals, w, label=f'NeuSym [{best_mode}]', color='#E91E63', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace('_','\n') for t in task_types], fontsize=7)
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("Exact Match", fontsize=10)
    ax.set_title("Per-Task-Type Performance\n(NeuSym vs both single paradigms)", fontsize=10)
    ax.legend(fontsize=9)
    ax.spines[['top','right']].set_visible(False)

    # Right: overall exact match comparison
    ax = axes[1]
    methods = ['Neural', 'Symbolic'] + [f'NeuSym\n[{m}]' for m in fusion_modes]
    vals = (
        [neural_results['exact_match'], symbolic_results['exact_match']] +
        [neusym_results_all[m]['exact_match'] for m in fusion_modes]
    )
    colors_bar = ['#2196F3','#4CAF50','#E91E63','#E91E63','#E91E63']
    alphas = [0.85, 0.85, 0.95, 0.65, 0.65]
    for i, (method, val, col, alph) in enumerate(zip(methods, vals, colors_bar, alphas)):
        ax.bar(i, val, color=col, alpha=alph, edgecolor='white')
        ax.text(i, val + 0.01, f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Exact Match Accuracy", fontsize=10)
    ax.set_title("Overall Accuracy\n(does fusion help?)", fontsize=10)
    ax.spines[['top','right']].set_visible(False)
    ax.axhline(max(neural_results['exact_match'], symbolic_results['exact_match']),
               color='red', linestyle='--', alpha=0.5, label='Best single paradigm')
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR + '//phase2_exp1_neusym.png',
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("\n  Figure saved: phase2_exp1_neusym.png")

    return save_data


if __name__ == "__main__":
    run_experiment()
