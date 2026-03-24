"""
PRAGMA Phase 9 — PRAGMA v1.4: Exact Match Verification
=======================================================
Phase 8 failure diagnosis revealed ONE root cause across ALL 8 failures:

  The system commits to a single-rule answer (high symbolic confidence)
  without verifying it's EXACTLY correct on training pairs.

  Examples:
    TRUE: color_replace → rot90cw
    Detected: rot90cw (conf=0.96) → WRONG because color step ignored
    
    TRUE: flipud → color_swap → fliplr  
    Detected: rot180 (conf=0.95) → WRONG because flipud∘fliplr≈rot180 but
              color_swap also transforms cells

  All 8 failures share this pattern: a high-confidence single rule
  approximates the true multi-step chain but is not exactly correct.

FIX — Exact Training Verification:
  Before returning ANY prediction, verify it exactly matches ALL
  training pair outputs. If any training pair fails exact match,
  fall through to the compositional decomposer.
  
  This is computationally cheap (just numpy array comparison) and
  eliminates the entire class of "good approximation, wrong answer" failures.

TARGET: 0.85+ on compositional benchmark

HOW TO RUN:
  python pragma/experiments/phase9_pragma_v14.py
"""

import os, sys, json, time
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
_RESULTS = os.path.join(_HERE, "results")
sys.path.insert(0, _ROOT)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

from pragma.benchmark.evaluator import evaluate_baseline
from pragma.baselines.symbolic import SymbolicBaseline, RULES
from pragma.baselines.neural import NeuralBaseline
from pragma.baselines.bayesian import BayesianBaseline
from pragma.experiments.phase4_pragma_v02 import (
    detect_full_causal_transform, apply_full_causal_transform
)
from pragma.experiments.phase6_compositional import (
    apply_chain, find_best_single_rule, generate_compositional_benchmark
)
from pragma.experiments.phase8_pragma_v13 import (
    generate_fixed_benchmark, make_compositional_task_fixed,
    PRAGMAv13
)
from pragma.experiments.phase7_pragma_v12 import INVERSE_PAIRS


def verify_on_training(prediction_fn, train_pairs):
    """
    Verify that a prediction function gives EXACT MATCH on all training pairs.
    prediction_fn: callable(input_grid) → output_grid
    Returns: (all_correct: bool, accuracy: float)
    """
    scores = []
    for inp, out in train_pairs:
        try:
            pred = prediction_fn(inp)
            if pred.shape == out.shape:
                scores.append(float(np.array_equal(pred, out)))
            else:
                scores.append(0.0)
        except Exception:
            scores.append(0.0)
    acc = float(np.mean(scores)) if scores else 0.0
    return acc == 1.0, acc


class PRAGMAv14(PRAGMAv13):
    """
    PRAGMA v1.4 — Exact Match Verification Gate.

    Single change over v1.3:
      Before committing to any single-rule prediction, verify it gives
      EXACT MATCH on ALL training pairs. If any pair fails, fall through
      to the compositional decomposer.

    This eliminates 'good approximation, wrong answer' failures where
    a geometric rule captures most of the transformation but misses
    the color step.
    """

    def _symbolic_applicator(self, rule_name, params):
        """Return a function that applies the named rule."""
        for rname, _, applicator in RULES:
            if rname == rule_name:
                return lambda inp, app=applicator, p=params: app(inp, p)
        return None

    def solve(self, task):
        bay_pred, unc = self._bay.solve_with_uncertainty(task)
        entropy = unc['entropy']

        # ── Tier 1: Symbolic — WITH exact match verification ──────────────
        if entropy < 0.5:
            sym_pred, expl = self._sym.solve_with_explanation(task)
            if sym_pred is not None and expl['confidence'] >= 0.85:
                # Verify exact match on training pairs
                rule_fn = self._symbolic_applicator(
                    expl['selected_rule'], expl.get('params', {}))
                if rule_fn:
                    all_correct, _ = verify_on_training(rule_fn, task.train_pairs)
                    if all_correct:
                        return sym_pred, 'tier1_symbolic_verified'
                    # Not exact — fall through to compositional

        # ── Tier 2: Causal + Symbolic — WITH exact match verification ─────
        if entropy < 2.0:
            hyps = detect_full_causal_transform(task.train_pairs)
            if hyps:
                pred = apply_full_causal_transform(task.test_input, hyps)
                if pred is not None:
                    return pred, 'tier2_causal'

            sym_pred, expl = self._sym.solve_with_explanation(task)
            if sym_pred is not None and expl['confidence'] >= 0.85:
                rule_fn = self._symbolic_applicator(
                    expl['selected_rule'], expl.get('params', {}))
                if rule_fn:
                    all_correct, _ = verify_on_training(rule_fn, task.train_pairs)
                    if all_correct:
                        return sym_pred, 'tier2_symbolic_verified'
                    # Not exact — fall through

        # ── Tier 3: Causal check ──────────────────────────────────────────
        hyps = detect_full_causal_transform(task.train_pairs)
        if hyps:
            pred = apply_full_causal_transform(task.test_input, hyps)
            if pred is not None:
                return pred, 'tier3_causal'

        # ── Compositional decomposition with consistency scoring ──────────
        valid_chains = self._collect_all_valid_chains(task.train_pairs, max_steps=3)
        if valid_chains:
            scored = [(chain, self._score_chain_consistency(chain, task.train_pairs))
                      for chain, _ in valid_chains]
            scored.sort(key=lambda x: -x[1])
            best_chain = scored[0][0]
            pred = apply_chain(task.test_input, best_chain)
            if pred is not None:
                n = best_chain['n_steps']
                return pred, f'consist_{n}step'

        # ── Neural fallback ────────────────────────────────────────────────
        neu_pred = self._neu.solve(task)
        if neu_pred is not None:
            return neu_pred, 'tier3_neural'
        return task.test_input.copy(), 'fallback'


def run_experiment():
    print("=" * 70)
    print("PRAGMA PHASE 9 — EXACT MATCH VERIFICATION GATE")
    print("=" * 70)
    print()
    print("Fix: Verify single-rule predictions on ALL training pairs before committing")
    print("     If any training pair fails exact match → compositional decomposer")
    print("Target: 0.85+ on compositional benchmark")
    print()

    # Reconstruct hybrid benchmark (same as Phase 8)
    old_tasks, old_chains = generate_compositional_benchmark(n_per_config=10, seed=42)
    old_2step = [t for t in old_tasks if t.difficulty in [2, 3]]

    rng_3step = np.random.default_rng(100)
    fixed_3step_tasks, fixed_3step_chains = [], []
    for _ in range(10):
        t, r = make_compositional_task_fixed(rng_3step, n_steps=3, n_train=5, difficulty=4)
        fixed_3step_tasks.append(t); fixed_3step_chains.append(r)

    hybrid_tasks = old_2step + fixed_3step_tasks
    all_chains = old_chains[:20] + fixed_3step_chains
    print(f"Using same hybrid benchmark as Phase 8: {len(hybrid_tasks)} tasks\n")

    # Phase 8 reference
    v13 = PRAGMAv13(beam_width=5)
    v13_preds, _ = v13.solve_batch(hybrid_tasks)
    v13_res = evaluate_baseline(hybrid_tasks, v13_preds)
    print(f"Phase 8 reference (v1.3): EM={v13_res['exact_match']:.3f}")

    # PRAGMA v1.4
    print("Running PRAGMA v1.4 (exact match verification)...")
    v14 = PRAGMAv14(beam_width=5)
    t0 = time.time()
    v14_preds, v14_tiers = v14.solve_batch(hybrid_tasks)
    elapsed = time.time() - t0
    v14_res = evaluate_baseline(hybrid_tasks, v14_preds)
    print(f"  EM={v14_res['exact_match']:.3f}  time={elapsed:.2f}s")
    print(f"  Tiers: {v14_tiers}")
    print()

    # Per task: show what changed
    print(f"  {'#':<3} {'Diff':<12} {'True chain':<35} {'v1.3':>6} {'v1.4':>6} {'Change'}")
    print("  " + "─" * 75)
    improvements, regressions = 0, 0
    for i, (task, chain) in enumerate(zip(hybrid_tasks, all_chains)):
        true_name = ' → '.join(r[0] for r in chain)
        v13_ok = np.array_equal(v13_preds[i], task.test_output) if v13_preds[i] is not None else False
        v14_ok = np.array_equal(v14_preds[i], task.test_output) if v14_preds[i] is not None else False
        diff_l = {2:'2-easy', 3:'2-med', 4:'3-hard'}[task.difficulty]
        if v13_ok != v14_ok:
            status = "✓ FIXED" if v14_ok else "✗ BROKE"
            if v14_ok: improvements += 1
            else: regressions += 1
            print(f"  {i+1:<3} {diff_l:<12} {true_name:<35} {'✓' if v13_ok else '✗':>6} {'✓' if v14_ok else '✗':>6}  {status}")
    if improvements + regressions == 0:
        print("  (no changes vs v1.3)")
    print()

    # Final summary
    print("=" * 70)
    print("COMPLETE PRAGMA PROGRESSION — ALL PHASES")
    print("=" * 70)
    all_scores = {
        'Single rules (v1.0)':          1.000,
        'Comp v1.0 (no decomp)':        0.267,
        'Comp v1.1 (exhaustive)':        0.600,
        'Comp v1.2/v1.3':               v13_res['exact_match'],
        'Comp v1.4 (verified)':          v14_res['exact_match'],
    }
    for name, score in all_scores.items():
        bar = '█' * int(score * 40)
        mark = " ◄ TARGET HIT" if score >= 0.85 else ""
        print(f"  {name:<32} {score:.3f}  {bar}{mark}")
    print()

    improvement = v14_res['exact_match'] - v13_res['exact_match']
    target_hit  = v14_res['exact_match'] >= 0.85
    print(f"  Improvement over v1.3:     {improvement:+.3f}")
    print(f"  New tasks fixed:           {improvements}")
    print(f"  Regressions:               {regressions}")
    print(f"  Target 0.85:               {'✓ HIT' if target_hit else '✗ MISSED'}")
    print()

    # Per-difficulty
    print(f"  {'Difficulty':<18} {'v1.3':>7} {'v1.4':>7} {'Delta':>7}")
    print("  " + "─" * 44)
    for diff, label in [(2,'2-step easy'), (3,'2-step medium'), (4,'3-step hard')]:
        v13d = v13_res['by_difficulty'].get(f'diff_{diff}', 0.0)
        v14d = v14_res['by_difficulty'].get(f'diff_{diff}', 0.0)
        flag = "★" if v14d - v13d > 0.05 else ""
        print(f"  {label:<18} {v13d:>7.3f} {v14d:>7.3f} {v14d-v13d:>+6.3f}  {flag}")
    print()
    print("SEND THIS ENTIRE OUTPUT BACK.")
    print("=" * 70)

    # Save
    os.makedirs(_RESULTS, exist_ok=True)
    with open(os.path.join(_RESULTS, 'phase9_pragma_v14.json'), 'w') as f:
        json.dump({'v13': v13_res, 'v14': v14_res,
                   'tiers': v14_tiers, 'target_hit': target_hit,
                   'improvements': improvements, 'regressions': regressions,
                   'all_scores': all_scores}, f, indent=2, default=str)

    _plot(v13_res, v14_res, v14_tiers, all_scores)


def _plot(v13_res, v14_res, v14_tiers, all_scores):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("PRAGMA Phase 9: Exact Match Verification Gate\n"
                 "Eliminating 'Good Approximation, Wrong Answer' Failures",
                 fontsize=13, fontweight='bold')

    # 1. Full progression
    ax = axes[0]
    names = list(all_scores.keys())
    vals  = [all_scores[n] for n in names]
    colors = ['#4CAF50','#90A4AE','#FF6F00','#00BCD4','#E91E63']
    bars = ax.bar(range(len(names)), vals, color=colors, alpha=0.9, edgecolor='white')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace(' ', '\n') for n in names], fontsize=7)
    ax.set_ylim(0, 1.15)
    ax.axhline(0.85, color='gold', linestyle='--', linewidth=2, label='Target 0.85')
    ax.axhline(1.0, color='green', linestyle=':', alpha=0.5)
    ax.legend(fontsize=9)
    for bar, val in zip(bars, vals):
        weight = 'bold' if val >= 0.85 else 'normal'
        ax.text(bar.get_x()+bar.get_width()/2, val+0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight=weight)
    ax.set_title("Complete Progression\n(Phases 1→9)", fontweight='bold')
    ax.set_ylabel("Exact Match")
    ax.spines[['top','right']].set_visible(False)

    # 2. Per-difficulty v1.3 vs v1.4
    ax = axes[1]
    diffs = [(2,'2-step\neasy'), (3,'2-step\nmed'), (4,'3-step\nhard')]
    x = np.arange(3); w = 0.35
    v13_d = [v13_res['by_difficulty'].get(f'diff_{d}',0) for d,_ in diffs]
    v14_d = [v14_res['by_difficulty'].get(f'diff_{d}',0) for d,_ in diffs]
    ax.bar(x-w/2, v13_d, w, label='v1.3 (P8)', color='#00BCD4', alpha=0.7)
    ax.bar(x+w/2, v14_d, w, label='v1.4 (P9)', color='#E91E63', alpha=0.95, edgecolor='white')
    ax.set_xticks(x); ax.set_xticklabels([l for _,l in diffs])
    ax.set_ylim(0, 1.15)
    ax.axhline(0.85, color='gold', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.legend(fontsize=10); ax.set_ylabel("Exact Match")
    ax.set_title("v1.3 vs v1.4 by Difficulty\n(verification gate impact)", fontweight='bold')
    ax.spines[['top','right']].set_visible(False)
    # Add value labels
    for bars in [ax.containers[0], ax.containers[1]]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x()+bar.get_width()/2, h+0.01,
                    f'{h:.2f}', ha='center', va='bottom', fontsize=9)

    # 3. Routing
    ax = axes[2]
    tiers = v14_tiers
    tlabels = list(tiers.keys())
    tvals   = list(tiers.values())
    tcolors = plt.cm.Set2(np.linspace(0, 1, len(tlabels)))
    bars = ax.bar(range(len(tvals)), tvals, color=tcolors, alpha=0.9, edgecolor='white')
    ax.set_xticks(range(len(tlabels)))
    ax.set_xticklabels([l.replace('_','\n') for l in tlabels], fontsize=7)
    for bar, val in zip(bars, tvals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                str(val), ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_ylabel("Tasks")
    ax.set_title("v1.4 Routing\n(verification gate changes flow)", fontweight='bold')
    ax.spines[['top','right']].set_visible(False)

    plt.tight_layout()
    path = os.path.join(_RESULTS, 'phase9_pragma_v14.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n  Figure saved: {path}")


if __name__ == "__main__":
    run_experiment()
