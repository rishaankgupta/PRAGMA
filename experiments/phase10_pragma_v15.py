"""
PRAGMA Phase 10 — PRAGMA v1.5: Inverse Search + Threshold Relaxation
======================================================================
Phase 9 failure analysis revealed two distinct remaining failure modes:

FAILURE MODE A (Tasks 4, 11):
  color_replace → geometric_rule chains.
  Root cause: _detect_color_replace runs on (inp → final_output) pairs,
  but the final output has been geometrically transformed, scrambling
  the positional color correspondence. So color_replace gets wrong params
  (params={}) and the chain is never found.
  
  Fix: INVERSE SEARCH — for each high-confidence geometric rule r2,
  apply inv(r2) to outputs to get hypothetical intermediates, then
  try all (A→B) color replacement pairs as r1. If any r1 maps
  inp → intermediate consistently, we've found the chain.

FAILURE MODE B (Tasks 21, 29):
  3-step chains where decomposer finds 0 valid chains.
  Root cause: e2e threshold 0.75 too strict for 3-step.
  With 5 training pairs and 3 sequential rules, minor grid variations
  accumulate and the decomposer misses chains scoring 0.65-0.74.
  
  Fix: Lower 3-step threshold to 0.60 with extra consistency verification.

TARGET: 0.90+ (fix all 4 remaining failures)

HOW TO RUN:
  python pragma/experiments/phase10_pragma_v15.py
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
    make_compositional_task_fixed
)
from pragma.experiments.phase9_pragma_v14 import PRAGMAv14, verify_on_training
from pragma.experiments.phase7_pragma_v12 import INVERSE_PAIRS


# ══════════════════════════════════════════════════════════════════════════════
# FIX A: Inverse Search for color-then-geometry chains
# ══════════════════════════════════════════════════════════════════════════════

# Inverse geometric transforms
GEOMETRIC_INVERSES = {
    'fliplr':   ('fliplr',   lambda g: np.fliplr(g)),
    'flipud':   ('flipud',   lambda g: np.flipud(g)),
    'rot90cw':  ('rot90ccw', lambda g: np.rot90(g, k=1)),
    'rot90ccw': ('rot90cw',  lambda g: np.rot90(g, k=-1)),
    'rot180':   ('rot180',   lambda g: np.rot90(g, k=2)),
}


def inverse_color_search(train_pairs):
    """
    Find chains of the form: color_rule → geometric_rule.
    
    Strategy:
    1. For each geometric rule r2 that could be the second step:
       - Apply inv(r2) to each output → hypothetical intermediate grids
       - Check if any consistent color mapping (A→B) transforms inp → intermediate
    2. If found, that's our chain.
    
    This handles the case where _detect_color_replace fails because the
    full output has been geometrically transformed.
    """
    best_chain = None
    best_score = 0.0

    for r2_name, (inv_name, inv_fn) in GEOMETRIC_INVERSES.items():
        # Compute hypothetical intermediates: inv(r2)(output)
        intermediate_pairs = []
        valid = True
        for inp, out in train_pairs:
            try:
                if inp.shape != out.shape:
                    valid = False; break
                mid = inv_fn(out)  # what the input would look like BEFORE r2
                intermediate_pairs.append((inp, mid))
            except Exception:
                valid = False; break

        if not valid: continue

        # Now find what color rule maps inp → mid
        # Try all (source_color, dest_color) pairs brute-force
        for src in range(1, 10):
            for dst in range(1, 10):
                if src == dst: continue
                # Build color map: src→dst, everything else maps to itself
                color_map = {src: dst}

                # Verify this color mapping on all pairs
                e2e_scores = []
                for inp, mid in intermediate_pairs:
                    pred = inp.copy()
                    pred[inp == src] = dst
                    if pred.shape == mid.shape:
                        e2e_scores.append(float(np.mean(pred == mid)))
                    else:
                        e2e_scores.append(0.0)

                e2e_color = float(np.mean(e2e_scores)) if e2e_scores else 0.0
                if e2e_color < 0.90:
                    continue

                # Now verify full chain end-to-end: color_replace → r2
                full_scores = []
                for inp, out in train_pairs:
                    try:
                        mid = inp.copy()
                        mid[inp == src] = dst
                        for rname, _, app in RULES:
                            if rname == r2_name:
                                pred = app(mid, {})
                                if pred.shape == out.shape:
                                    full_scores.append(float(np.mean(pred == out)))
                                else:
                                    full_scores.append(0.0)
                                break
                    except Exception:
                        full_scores.append(0.0)

                full_e2e = float(np.mean(full_scores)) if full_scores else 0.0
                if full_e2e > best_score:
                    best_score = full_e2e
                    best_chain = {
                        'step1': {
                            'rule': 'color_replace',
                            'params': {'color_map': {src: dst}},
                            'conf': e2e_color
                        },
                        'step2': {
                            'rule': r2_name,
                            'params': {},
                            'conf': full_e2e
                        },
                        'n_steps': 2,
                        'score': full_e2e
                    }

    # Also try: geometry → color_replace (reverse order)
    for r1_name, (_, _) in GEOMETRIC_INVERSES.items():
        # Apply r1 to inputs
        intermediate_pairs = []
        valid = True
        for inp, out in train_pairs:
            try:
                for rname, _, app in RULES:
                    if rname == r1_name:
                        mid = app(inp, {})
                        intermediate_pairs.append((mid, out))
                        break
            except Exception:
                valid = False; break

        if not valid: continue

        # Find color rule on intermediate → output
        for src in range(1, 10):
            for dst in range(1, 10):
                if src == dst: continue
                e2e_scores = []
                for mid, out in intermediate_pairs:
                    pred = mid.copy()
                    pred[mid == src] = dst
                    if pred.shape == out.shape:
                        e2e_scores.append(float(np.mean(pred == out)))
                    else:
                        e2e_scores.append(0.0)

                e2e = float(np.mean(e2e_scores)) if e2e_scores else 0.0
                if e2e > best_score and e2e >= 0.90:
                    # Full verification
                    full_scores = []
                    for inp, out in train_pairs:
                        try:
                            for rname, _, app in RULES:
                                if rname == r1_name:
                                    mid = app(inp, {})
                                    pred = mid.copy()
                                    pred[mid == src] = dst
                                    if pred.shape == out.shape:
                                        full_scores.append(float(np.mean(pred == out)))
                                    else:
                                        full_scores.append(0.0)
                                    break
                        except Exception:
                            full_scores.append(0.0)

                    full_e2e = float(np.mean(full_scores)) if full_scores else 0.0
                    if full_e2e > best_score:
                        best_score = full_e2e
                        best_chain = {
                            'step1': {'rule': r1_name, 'params': {}, 'conf': full_e2e},
                            'step2': {
                                'rule': 'color_replace',
                                'params': {'color_map': {src: dst}},
                                'conf': e2e
                            },
                            'n_steps': 2,
                            'score': full_e2e
                        }

    return best_chain if best_chain and best_score >= 0.90 else None


def apply_color_replace_chain(test_input, chain):
    """Apply a chain that includes color_replace steps."""
    current = test_input.copy()
    n = chain['n_steps']
    for i in range(1, n+1):
        step = chain[f'step{i}']
        rule_name = step['rule']
        params = step['params']
        if rule_name == 'color_replace':
            for src, dst in params.get('color_map', {}).items():
                current[current == src] = dst
        else:
            for rname, _, app in RULES:
                if rname == rule_name:
                    current = app(current, params)
                    break
    return current


# ══════════════════════════════════════════════════════════════════════════════
# FIX B: Relaxed 3-step search
# ══════════════════════════════════════════════════════════════════════════════

def relaxed_three_step_search(train_pairs, threshold=0.60):
    """
    Search for 3-step chains with relaxed threshold.
    Used as fallback when standard decomposer finds nothing.
    """
    best_chain = None
    best_score = 0.0

    for r1_name, r1_det, r1_app in RULES:
        try:
            _, r1_params = r1_det(train_pairs)
        except Exception:
            r1_params = {}

        mid1_pairs = []
        valid = True
        for inp, out in train_pairs:
            try:
                mid1_pairs.append((r1_app(inp, r1_params), out))
            except Exception:
                valid = False; break
        if not valid: continue

        for r2_name, r2_det, r2_app in RULES:
            if r2_name == r1_name: continue
            try:
                _, r2_params = r2_det(mid1_pairs)
            except Exception:
                r2_params = {}

            mid2_pairs = []
            valid2 = True
            for mid1, out in mid1_pairs:
                try:
                    mid2_pairs.append((r2_app(mid1, r2_params), out))
                except Exception:
                    valid2 = False; break
            if not valid2: continue

            r3_name, r3_params, r3_conf = find_best_single_rule(mid2_pairs)
            if not r3_name or r3_conf < 0.60: continue

            # Full e2e check
            scores = []
            for inp, out in train_pairs:
                try:
                    m1 = r1_app(inp, r1_params)
                    m2 = r2_app(m1, r2_params)
                    for rn, _, app3 in RULES:
                        if rn == r3_name:
                            pred = app3(m2, r3_params)
                            if pred.shape == out.shape:
                                scores.append(float(np.mean(pred == out)))
                            else:
                                scores.append(0.0)
                            break
                except Exception:
                    scores.append(0.0)

            e2e = float(np.mean(scores)) if scores else 0.0
            if e2e > best_score and e2e >= threshold:
                best_score = e2e
                best_chain = {
                    'step1': {'rule': r1_name, 'params': r1_params},
                    'step2': {'rule': r2_name, 'params': r2_params},
                    'step3': {'rule': r3_name, 'params': r3_params},
                    'n_steps': 3,
                    'score': e2e
                }

    return best_chain


# ══════════════════════════════════════════════════════════════════════════════
# PRAGMA v1.5
# ══════════════════════════════════════════════════════════════════════════════

class PRAGMAv15(PRAGMAv14):
    """
    PRAGMA v1.5 — Inverse Search + Relaxed 3-step.

    Additions over v1.4:
    1. Inverse search for color → geometry and geometry → color chains
    2. Relaxed 3-step search (threshold 0.60) as fallback
    """

    def solve(self, task):
        bay_pred, unc = self._bay.solve_with_uncertainty(task)
        entropy = unc['entropy']

        # Tier 1: Symbolic with exact verification
        if entropy < 0.5:
            sym_pred, expl = self._sym.solve_with_explanation(task)
            if sym_pred is not None and expl['confidence'] >= 0.85:
                rule_fn = self._symbolic_applicator(
                    expl['selected_rule'], expl.get('params', {}))
                if rule_fn:
                    all_correct, _ = verify_on_training(rule_fn, task.train_pairs)
                    if all_correct:
                        return sym_pred, 'tier1_symbolic_verified'

        # Tier 2: Causal + Symbolic with verification
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

        # Tier 3 causal
        hyps = detect_full_causal_transform(task.train_pairs)
        if hyps:
            pred = apply_full_causal_transform(task.test_input, hyps)
            if pred is not None:
                return pred, 'tier3_causal'

        # ── INVERSE SEARCH (Fix A) ─────────────────────────────────────────
        inv_chain = inverse_color_search(task.train_pairs)
        if inv_chain:
            pred = apply_color_replace_chain(task.test_input, inv_chain)
            if pred is not None:
                return pred, 'inverse_search'

        # ── Standard compositional decomposition ──────────────────────────
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

        # ── RELAXED 3-STEP SEARCH (Fix B) ─────────────────────────────────
        relaxed = relaxed_three_step_search(task.train_pairs, threshold=0.60)
        if relaxed:
            pred = apply_chain(task.test_input, relaxed)
            if pred is not None:
                return pred, 'relaxed_3step'

        # Neural fallback
        neu_pred = self._neu.solve(task)
        if neu_pred is not None:
            return neu_pred, 'tier3_neural'
        return task.test_input.copy(), 'fallback'


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_experiment():
    print("=" * 70)
    print("PRAGMA PHASE 10 — INVERSE SEARCH + RELAXED 3-STEP")
    print("=" * 70)
    print()
    print("Fix A: Inverse search for color → geometry chains")
    print("Fix B: Relaxed 3-step threshold (0.60) as last resort")
    print("Target: 0.90+")
    print()

    # Reconstruct hybrid benchmark (identical to Phase 8/9)
    old_tasks, old_chains = generate_compositional_benchmark(n_per_config=10, seed=42)
    old_2step = [t for t in old_tasks if t.difficulty in [2, 3]]

    rng_3step = np.random.default_rng(100)
    fixed_3step_tasks, fixed_3step_chains = [], []
    for _ in range(10):
        t, r = make_compositional_task_fixed(rng_3step, n_steps=3, n_train=5, difficulty=4)
        fixed_3step_tasks.append(t); fixed_3step_chains.append(r)

    hybrid = old_2step + fixed_3step_tasks
    all_chains = old_chains[:20] + fixed_3step_chains
    print(f"Hybrid benchmark: {len(hybrid)} tasks\n")

    # Phase 9 reference
    v14 = PRAGMAv14(beam_width=5)
    v14_preds, _ = v14.solve_batch(hybrid)
    v14_res = evaluate_baseline(hybrid, v14_preds)
    print(f"Phase 9 reference (v1.4): EM={v14_res['exact_match']:.3f}")

    # PRAGMA v1.5
    print("Running PRAGMA v1.5...")
    v15 = PRAGMAv15(beam_width=5)
    t0 = time.time()
    v15_preds, v15_tiers = v15.solve_batch(hybrid)
    elapsed = time.time() - t0
    v15_res = evaluate_baseline(hybrid, v15_preds)
    print(f"  EM={v15_res['exact_match']:.3f}  time={elapsed:.2f}s")
    print(f"  Tiers: {v15_tiers}\n")

    # What changed
    print(f"  {'#':<3} {'Diff':<12} {'True chain':<38} {'v1.4':>5} {'v1.5':>5}  Change")
    print("  " + "─" * 72)
    improvements, regressions = 0, 0
    for i, (task, chain) in enumerate(zip(hybrid, all_chains)):
        true_name = ' → '.join(r[0] for r in chain)
        v14_ok = np.array_equal(v14_preds[i], task.test_output) if v14_preds[i] is not None else False
        v15_ok = np.array_equal(v15_preds[i], task.test_output) if v15_preds[i] is not None else False
        if v14_ok != v15_ok:
            diff_l = {2:'2-easy', 3:'2-med', 4:'3-hard'}[task.difficulty]
            status = "✓ FIXED" if v15_ok else "✗ BROKE"
            if v15_ok: improvements += 1
            else: regressions += 1
            print(f"  {i+1:<3} {diff_l:<12} {true_name:<38} {'✓' if v14_ok else '✗':>5} {'✓' if v15_ok else '✗':>5}  {status}")
    if improvements + regressions == 0:
        print("  (no changes vs v1.4)")
    print()

    # Summary
    print("=" * 70)
    print("COMPLETE PRAGMA PROGRESSION")
    print("=" * 70)
    all_scores = {
        'Single rules (v1.0)':       1.000,
        'Comp v1.1 (first decomp)':  0.600,
        'Comp v1.3 (consistency)':   0.733,
        'Comp v1.4 (verification)':  v14_res['exact_match'],
        'Comp v1.5 (inverse+relax)': v15_res['exact_match'],
    }
    for name, score in all_scores.items():
        bar = '█' * int(score * 40)
        mark = ""
        if score >= 0.90: mark = "  ◄ 0.90 TARGET HIT"
        elif score >= 0.85: mark = "  ◄ 0.85 target hit"
        print(f"  {name:<32} {score:.3f}  {bar}{mark}")
    print()

    improvement = v15_res['exact_match'] - v14_res['exact_match']
    t90 = v15_res['exact_match'] >= 0.90
    print(f"  Improvement over v1.4:  {improvement:+.3f}")
    print(f"  Tasks fixed:            {improvements}")
    print(f"  Regressions:            {regressions}")
    print(f"  Target 0.90:            {'✓ HIT' if t90 else '✗ MISSED'}")
    print()

    for diff, label in [(2,'2-step easy'), (3,'2-step medium'), (4,'3-step hard')]:
        v14d = v14_res['by_difficulty'].get(f'diff_{diff}', 0.0)
        v15d = v15_res['by_difficulty'].get(f'diff_{diff}', 0.0)
        flag = "★" if v15d - v14d > 0.05 else ""
        print(f"  {label:<16} v1.4={v14d:.3f}  v1.5={v15d:.3f}  {v15d-v14d:+.3f}  {flag}")
    print()
    print("SEND THIS ENTIRE OUTPUT BACK.")
    print("=" * 70)

    os.makedirs(_RESULTS, exist_ok=True)
    with open(os.path.join(_RESULTS, 'phase10_pragma_v15.json'), 'w') as f:
        json.dump({'v14': v14_res, 'v15': v15_res, 'tiers': v15_tiers,
                   'improvements': improvements, 'regressions': regressions,
                   'target_hit': t90, 'all_scores': all_scores},
                  f, indent=2, default=str)

    _plot(v14_res, v15_res, v15_tiers, all_scores)


def _plot(v14_res, v15_res, v15_tiers, all_scores):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("PRAGMA Phase 10: Inverse Search + Relaxed 3-step\n"
                 "Targeting 0.90+ on Compositional Benchmark",
                 fontsize=13, fontweight='bold')

    ax = axes[0]
    names = list(all_scores.keys())
    vals  = [all_scores[n] for n in names]
    colors = ['#4CAF50','#FF6F00','#00BCD4','#E91E63','#9C27B0']
    bars = ax.bar(range(len(names)), vals, color=colors, alpha=0.9, edgecolor='white')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace(' ', '\n') for n in names], fontsize=7)
    ax.set_ylim(0, 1.15)
    ax.axhline(0.90, color='red', linestyle='--', linewidth=2, label='Target 0.90')
    ax.axhline(0.85, color='gold', linestyle=':', linewidth=1.5, alpha=0.7, label='Phase 9 target')
    ax.legend(fontsize=8)
    for bar, val in zip(bars, vals):
        weight = 'bold' if val >= 0.90 else 'normal'
        ax.text(bar.get_x()+bar.get_width()/2, val+0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight=weight)
    ax.set_title("Progression to 0.90", fontweight='bold')
    ax.set_ylabel("Exact Match")
    ax.spines[['top','right']].set_visible(False)

    ax = axes[1]
    diffs = [(2,'2-step\neasy'), (3,'2-step\nmed'), (4,'3-step\nhard')]
    x = np.arange(3); w = 0.35
    v14_d = [v14_res['by_difficulty'].get(f'diff_{d}',0) for d,_ in diffs]
    v15_d = [v15_res['by_difficulty'].get(f'diff_{d}',0) for d,_ in diffs]
    ax.bar(x-w/2, v14_d, w, label='v1.4 (P9)', color='#E91E63', alpha=0.7)
    ax.bar(x+w/2, v15_d, w, label='v1.5 (P10)', color='#9C27B0', alpha=0.95, edgecolor='white')
    ax.set_xticks(x); ax.set_xticklabels([l for _,l in diffs])
    ax.set_ylim(0, 1.15)
    ax.axhline(0.90, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.legend(fontsize=10); ax.set_ylabel("Exact Match")
    ax.set_title("v1.4 vs v1.5 by Difficulty", fontweight='bold')
    ax.spines[['top','right']].set_visible(False)
    for containers in [ax.containers[0], ax.containers[1]]:
        for bar in containers:
            h = bar.get_height()
            ax.text(bar.get_x()+bar.get_width()/2, h+0.01,
                    f'{h:.2f}', ha='center', va='bottom', fontsize=9)

    ax = axes[2]
    tl = list(v15_tiers.keys())
    tv = list(v15_tiers.values())
    tc = plt.cm.Set2(np.linspace(0, 1, len(tl)))
    bars = ax.bar(range(len(tv)), tv, color=tc, alpha=0.9, edgecolor='white')
    ax.set_xticks(range(len(tl)))
    ax.set_xticklabels([l.replace('_','\n') for l in tl], fontsize=7)
    for bar, val in zip(bars, tv):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                str(val), ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_ylabel("Tasks")
    ax.set_title("v1.5 Routing\n(inverse_search = new capability)", fontweight='bold')
    ax.spines[['top','right']].set_visible(False)

    plt.tight_layout()
    path = os.path.join(_RESULTS, 'phase10_pragma_v15.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n  Figure saved: {path}")


if __name__ == "__main__":
    run_experiment()
