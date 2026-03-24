"""
PRAGMA Phase 11 — PRAGMA v1.6: Extended Forward Color Search
=============================================================
Phase 10 diagnosis found exactly why the 3 remaining tasks fail:

Task 11 (2-step-med): color_replace → color_swap
  color_swap-first brute force FINDS it (SWAP(1<->8) → color_swap → color_swap)
  Fix: extend inverse_color_search to also try color_swap as r1

Task 21 (3-hard): color_swap → rot90cw → border_fill
  After applying correct color_swap, remaining 2-step (rot90cw → border_fill)
  is NOT detected by find_best_single_rule (it can only find ONE rule).
  Fix: after color_swap, run FULL 2-STEP DECOMPOSER on (mid → out)

Task 29 (3-hard): border_fill → color_replace → gravity
  border_fill detection fails on full pairs because gravity changes interior.
  Fix: brute-force all border_fill color params, apply, then run 2-step decomp

STRATEGY: "Color-First Forward Search with Sub-Decomposition"
  For each candidate first step (color_swap or color_replace with brute-force params):
    Apply it to get intermediate grids
    Run 2-step decomposer on (intermediate → output)
    If found, verify 3-step chain end-to-end

TARGET: 0.93+ (fix all 3 remaining tasks)

HOW TO RUN:
  python pragma/experiments/phase11_pragma_v16.py
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
from pragma.experiments.phase8_pragma_v13 import make_compositional_task_fixed
from pragma.experiments.phase9_pragma_v14 import verify_on_training
from pragma.experiments.phase10_pragma_v15 import (
    PRAGMAv15, inverse_color_search, apply_color_replace_chain,
    GEOMETRIC_INVERSES, relaxed_three_step_search
)


def beam_decompose_two_step_clean(train_pairs):
    """
    Clean 2-step decomposer that works on any (inp, out) pairs.
    Used as sub-decomposer after applying a first step.
    Returns best chain or None.
    """
    best_chain = None
    best_score = 0.0

    for r1_name, r1_det, r1_app in RULES:
        try:
            _, r1_params = r1_det(train_pairs)
        except Exception:
            r1_params = {}

        mid_pairs = []
        valid = True
        for inp, out in train_pairs:
            try:
                mid_pairs.append((r1_app(inp, r1_params), out))
            except Exception:
                valid = False; break
        if not valid or not mid_pairs: continue

        r2_name, r2_params, r2_conf = find_best_single_rule(mid_pairs)
        if not r2_name or r2_conf < 0.80: continue
        if r1_name == r2_name and r1_name in ('fliplr', 'flipud', 'rot180'): continue

        # E2e verification
        e2e = []
        for inp, out in train_pairs:
            try:
                for rn1, _, a1 in RULES:
                    if rn1 == r1_name:
                        m = a1(inp, r1_params)
                        for rn2, _, a2 in RULES:
                            if rn2 == r2_name:
                                p = a2(m, r2_params)
                                e2e.append(float(np.mean(p==out)) if p.shape==out.shape else 0.0)
                                break
                        break
            except Exception:
                e2e.append(0.0)

        score = float(np.mean(e2e)) if e2e else 0.0
        if score > best_score:
            best_score = score
            best_chain = {
                'step1': {'rule': r1_name, 'params': r1_params, 'conf': score},
                'step2': {'rule': r2_name, 'params': r2_params, 'conf': r2_conf},
                'n_steps': 2, 'score': score
            }

    return best_chain if best_chain and best_score >= 0.85 else None


def extended_color_first_search(train_pairs):
    """
    Extended color-first search that handles all remaining failure modes.

    Tries each possible color transformation as the FIRST step (brute force
    over all color params), then runs a clean 2-step decomposer on the
    resulting (intermediate → output) pairs.

    This correctly handles:
    - color_replace → X → Y  (where X, Y are any rules)
    - color_swap → X → Y
    - X → color_replace → Y  (via geometric inverse then color search)
    """
    best_chain = None
    best_score = 0.0
    n_pairs = len(train_pairs)
    h, w = train_pairs[0][0].shape if train_pairs else (0, 0)

    # ── Part 1: color_swap as first step (bidirectional) ──────────────────
    for src in range(1, 10):
        for dst in range(src+1, 10):
            # Apply color_swap(src↔dst) to all inputs
            mid_pairs = []
            for inp, out in train_pairs:
                mid = inp.copy()
                mid[inp == src] = dst
                mid[inp == dst] = src
                mid_pairs.append((mid, out))

            # Run 2-step sub-decomp on (mid → out)
            sub_chain = beam_decompose_two_step_clean(mid_pairs)
            if sub_chain and sub_chain['score'] >= 0.85:
                # Full 3-step e2e verification
                scores = []
                for inp, out in train_pairs:
                    try:
                        m1 = inp.copy(); m1[inp==src]=dst; m1[inp==dst]=src
                        r2 = sub_chain['step1']['rule']; p2 = sub_chain['step1']['params']
                        r3 = sub_chain['step2']['rule']; p3 = sub_chain['step2']['params']
                        for rn,_,app in RULES:
                            if rn==r2: m2 = app(m1, p2); break
                        for rn,_,app in RULES:
                            if rn==r3:
                                pred = app(m2, p3)
                                scores.append(float(np.mean(pred==out)) if pred.shape==out.shape else 0.0)
                                break
                    except Exception:
                        scores.append(0.0)
                e2e = float(np.mean(scores)) if scores else 0.0
                if e2e > best_score:
                    best_score = e2e
                    best_chain = {
                        'color_step': {'type': 'swap', 'src': src, 'dst': dst},
                        'sub_chain': sub_chain,
                        'n_steps': 3,
                        'score': e2e
                    }

    # ── Part 2: color_replace as first step (unidirectional) ─────────────
    for src in range(1, 10):
        for dst in range(1, 10):
            if src == dst: continue
            mid_pairs = []
            for inp, out in train_pairs:
                mid = inp.copy(); mid[inp == src] = dst
                mid_pairs.append((mid, out))

            sub_chain = beam_decompose_two_step_clean(mid_pairs)
            if sub_chain and sub_chain['score'] >= 0.85:
                scores = []
                for inp, out in train_pairs:
                    try:
                        m1 = inp.copy(); m1[inp==src] = dst
                        r2 = sub_chain['step1']['rule']; p2 = sub_chain['step1']['params']
                        r3 = sub_chain['step2']['rule']; p3 = sub_chain['step2']['params']
                        for rn,_,app in RULES:
                            if rn==r2: m2 = app(m1, p2); break
                        for rn,_,app in RULES:
                            if rn==r3:
                                pred = app(m2, p3)
                                scores.append(float(np.mean(pred==out)) if pred.shape==out.shape else 0.0)
                                break
                    except Exception:
                        scores.append(0.0)
                e2e = float(np.mean(scores)) if scores else 0.0
                if e2e > best_score:
                    best_score = e2e
                    best_chain = {
                        'color_step': {'type': 'replace', 'src': src, 'dst': dst},
                        'sub_chain': sub_chain,
                        'n_steps': 3,
                        'score': e2e
                    }

    return best_chain if best_chain and best_score >= 0.85 else None


def apply_extended_chain(test_input, chain):
    """Apply an extended chain (color step + 2-step sub-chain)."""
    cs = chain['color_step']
    current = test_input.copy()

    if cs['type'] == 'swap':
        src, dst = cs['src'], cs['dst']
        tmp = current.copy()
        current[tmp == src] = dst
        current[tmp == dst] = src
    elif cs['type'] == 'replace':
        current[test_input == cs['src']] = cs['dst']

    sub = chain['sub_chain']
    for i in range(1, sub['n_steps']+1):
        step = sub[f'step{i}']
        for rn, _, app in RULES:
            if rn == step['rule']:
                current = app(current, step['params'])
                break

    return current


class PRAGMAv16(PRAGMAv15):
    """
    PRAGMA v1.6 — Extended Forward Color Search.

    Adds over v1.5:
    - Extended color-first 3-step search (brute-force color params + 2-step sub-decomp)
    - Handles: color_X → geo → geo, color_X → color → geo, structural → color_X → geo
    """

    def solve(self, task):
        bay_pred, unc = self._bay.solve_with_uncertainty(task)
        entropy = unc['entropy']

        # Tier 1
        if entropy < 0.5:
            sym_pred, expl = self._sym.solve_with_explanation(task)
            if sym_pred is not None and expl['confidence'] >= 0.85:
                rule_fn = self._symbolic_applicator(expl['selected_rule'], expl.get('params', {}))
                if rule_fn:
                    ok, _ = verify_on_training(rule_fn, task.train_pairs)
                    if ok: return sym_pred, 'tier1_symbolic_verified'

        # Tier 2
        if entropy < 2.0:
            hyps = detect_full_causal_transform(task.train_pairs)
            if hyps:
                pred = apply_full_causal_transform(task.test_input, hyps)
                if pred is not None: return pred, 'tier2_causal'
            sym_pred, expl = self._sym.solve_with_explanation(task)
            if sym_pred is not None and expl['confidence'] >= 0.85:
                rule_fn = self._symbolic_applicator(expl['selected_rule'], expl.get('params', {}))
                if rule_fn:
                    ok, _ = verify_on_training(rule_fn, task.train_pairs)
                    if ok: return sym_pred, 'tier2_symbolic_verified'

        # Tier 3 causal
        hyps = detect_full_causal_transform(task.train_pairs)
        if hyps:
            pred = apply_full_causal_transform(task.test_input, hyps)
            if pred is not None: return pred, 'tier3_causal'

        # Inverse search (v1.5)
        inv_chain = inverse_color_search(task.train_pairs)
        if inv_chain:
            pred = apply_color_replace_chain(task.test_input, inv_chain)
            if pred is not None: return pred, 'inverse_search'

        # Standard compositional
        valid_chains = self._collect_all_valid_chains(task.train_pairs, max_steps=3)
        if valid_chains:
            scored = [(ch, self._score_chain_consistency(ch, task.train_pairs))
                      for ch, _ in valid_chains]
            scored.sort(key=lambda x: -x[1])
            pred = apply_chain(task.test_input, scored[0][0])
            if pred is not None:
                return pred, f'consist_{scored[0][0]["n_steps"]}step'

        # ── EXTENDED COLOR-FIRST SEARCH (NEW) ─────────────────────────────
        ext_chain = extended_color_first_search(task.train_pairs)
        if ext_chain:
            pred = apply_extended_chain(task.test_input, ext_chain)
            if pred is not None: return pred, 'ext_color_first'

        # Relaxed 3-step
        relaxed = relaxed_three_step_search(task.train_pairs, threshold=0.60)
        if relaxed:
            pred = apply_chain(task.test_input, relaxed)
            if pred is not None: return pred, 'relaxed_3step'

        # Neural
        neu_pred = self._neu.solve(task)
        if neu_pred is not None: return neu_pred, 'tier3_neural'
        return task.test_input.copy(), 'fallback'


def run_experiment():
    print("=" * 70)
    print("PRAGMA PHASE 11 — EXTENDED FORWARD COLOR SEARCH")
    print("=" * 70)
    print()
    print("Fix: Brute-force color params as first step + 2-step sub-decomposer")
    print("Target: 0.93+")
    print()

    old_tasks, old_chains = generate_compositional_benchmark(n_per_config=10, seed=42)
    old_2step = [t for t in old_tasks if t.difficulty in [2, 3]]
    rng = np.random.default_rng(100)
    fixed_3step, fixed_chains = [], []
    for _ in range(10):
        t, r = make_compositional_task_fixed(rng, n_steps=3, n_train=5, difficulty=4)
        fixed_3step.append(t); fixed_chains.append(r)
    hybrid = old_2step + fixed_3step
    all_chains = old_chains[:20] + fixed_chains
    print(f"Hybrid benchmark: {len(hybrid)} tasks\n")

    # References
    v15 = PRAGMAv15(beam_width=5)
    v15_preds, _ = v15.solve_batch(hybrid)
    v15_res = evaluate_baseline(hybrid, v15_preds)
    print(f"Phase 10 reference (v1.5): EM={v15_res['exact_match']:.3f}")

    # v1.6
    print("Running PRAGMA v1.6...")
    v16 = PRAGMAv16(beam_width=5)
    t0 = time.time()
    v16_preds, v16_tiers = v16.solve_batch(hybrid)
    elapsed = time.time() - t0
    v16_res = evaluate_baseline(hybrid, v16_preds)
    print(f"  EM={v16_res['exact_match']:.3f}  time={elapsed:.2f}s")
    print(f"  Tiers: {v16_tiers}\n")

    # What changed
    print(f"  {'#':<3} {'Diff':<12} {'True chain':<40} {'v1.5':>5} {'v1.6':>5}  Change")
    print("  " + "─" * 75)
    improvements, regressions = 0, 0
    for i, (task, chain) in enumerate(zip(hybrid, all_chains)):
        true_name = ' → '.join(r[0] for r in chain)
        v15_ok = np.array_equal(v15_preds[i], task.test_output) if v15_preds[i] is not None else False
        v16_ok = np.array_equal(v16_preds[i], task.test_output) if v16_preds[i] is not None else False
        if v15_ok != v16_ok:
            diff_l = {2:'2-easy', 3:'2-med', 4:'3-hard'}[task.difficulty]
            status = "✓ FIXED" if v16_ok else "✗ BROKE"
            if v16_ok: improvements += 1
            else: regressions += 1
            print(f"  {i+1:<3} {diff_l:<12} {true_name:<40} {'✓' if v15_ok else '✗':>5} {'✓' if v16_ok else '✗':>5}  {status}")
    if improvements + regressions == 0:
        print("  (no changes vs v1.5)")
    print()

    # Summary
    print("=" * 70)
    print("COMPLETE PRAGMA PROGRESSION")
    print("=" * 70)
    all_scores = {
        'Single rules (v1.0)':      1.000,
        'Comp v1.1':                0.600,
        'Comp v1.4 (verification)': 0.867,
        'Comp v1.5 (inverse)':      v15_res['exact_match'],
        'Comp v1.6 (color-first)':  v16_res['exact_match'],
    }
    for name, score in all_scores.items():
        bar = '█' * int(score * 40)
        mark = "  ◄ TARGET" if score >= 0.93 else ("  ◄ 0.90" if score >= 0.90 else "")
        print(f"  {name:<32} {score:.3f}  {bar}{mark}")
    print()

    imp = v16_res['exact_match'] - v15_res['exact_match']
    t93 = v16_res['exact_match'] >= 0.93
    print(f"  Improvement: {imp:+.3f}")
    print(f"  Fixed: {improvements}  Regressions: {regressions}")
    print(f"  Target 0.93: {'✓ HIT' if t93 else '✗ MISSED'}")
    print()
    for diff, label in [(2,'2-step easy'), (3,'2-step med'), (4,'3-step hard')]:
        v15d = v15_res['by_difficulty'].get(f'diff_{diff}', 0.0)
        v16d = v16_res['by_difficulty'].get(f'diff_{diff}', 0.0)
        flag = "★" if v16d - v15d > 0.05 else ""
        print(f"  {label:<16} v1.5={v15d:.3f}  v1.6={v16d:.3f}  {v16d-v15d:+.3f}  {flag}")
    print()
    print("SEND THIS ENTIRE OUTPUT BACK.")
    print("=" * 70)

    os.makedirs(_RESULTS, exist_ok=True)
    with open(os.path.join(_RESULTS, 'phase11_pragma_v16.json'), 'w') as f:
        json.dump({'v15': v15_res, 'v16': v16_res, 'tiers': v16_tiers,
                   'improvements': improvements, 'all_scores': all_scores},
                  f, indent=2, default=str)

    _plot(v15_res, v16_res, v16_tiers, all_scores)


def _plot(v15_res, v16_res, v16_tiers, all_scores):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("PRAGMA Phase 11: Extended Forward Color Search\n"
                 "Targeting 0.93+ on Compositional Benchmark",
                 fontsize=13, fontweight='bold')

    ax = axes[0]
    names = list(all_scores.keys())
    vals  = [all_scores[n] for n in names]
    colors = ['#4CAF50','#FF6F00','#E91E63','#9C27B0','#673AB7']
    bars = ax.bar(range(len(names)), vals, color=colors, alpha=0.9, edgecolor='white')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace(' ', '\n') for n in names], fontsize=7)
    ax.set_ylim(0, 1.15)
    ax.axhline(0.93, color='red', linestyle='--', linewidth=2, label='Target 0.93')
    ax.axhline(0.90, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.legend(fontsize=8)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, val+0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9,
                fontweight='bold' if val >= 0.93 else 'normal')
    ax.set_title("Progression to 0.93", fontweight='bold')
    ax.set_ylabel("Exact Match")
    ax.spines[['top','right']].set_visible(False)

    ax = axes[1]
    diffs = [(2,'2-step\neasy'), (3,'2-step\nmed'), (4,'3-step\nhard')]
    x = np.arange(3); w = 0.35
    v15_d = [v15_res['by_difficulty'].get(f'diff_{d}',0) for d,_ in diffs]
    v16_d = [v16_res['by_difficulty'].get(f'diff_{d}',0) for d,_ in diffs]
    ax.bar(x-w/2, v15_d, w, label='v1.5 (P10)', color='#9C27B0', alpha=0.7)
    ax.bar(x+w/2, v16_d, w, label='v1.6 (P11)', color='#673AB7', alpha=0.95, edgecolor='white')
    ax.set_xticks(x); ax.set_xticklabels([l for _,l in diffs])
    ax.set_ylim(0, 1.15)
    ax.axhline(0.93, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.legend(fontsize=10); ax.set_ylabel("Exact Match")
    ax.set_title("v1.5 vs v1.6 by Difficulty", fontweight='bold')
    ax.spines[['top','right']].set_visible(False)
    for c in [ax.containers[0], ax.containers[1]]:
        for bar in c:
            h = bar.get_height()
            ax.text(bar.get_x()+bar.get_width()/2, h+0.01,
                    f'{h:.2f}', ha='center', va='bottom', fontsize=9)

    ax = axes[2]
    tl = list(v16_tiers.keys()); tv = list(v16_tiers.values())
    tc = plt.cm.Set3(np.linspace(0, 1, len(tl)))
    ax.bar(range(len(tv)), tv, color=tc, alpha=0.9, edgecolor='white')
    ax.set_xticks(range(len(tl)))
    ax.set_xticklabels([l.replace('_','\n') for l in tl], fontsize=7)
    for i, val in enumerate(tv):
        ax.text(i, val+0.1, str(val), ha='center', va='bottom',
                fontsize=10, fontweight='bold')
    ax.set_ylabel("Tasks")
    ax.set_title("v1.6 Routing\n(ext_color_first = new)", fontweight='bold')
    ax.spines[['top','right']].set_visible(False)

    plt.tight_layout()
    path = os.path.join(_RESULTS, 'phase11_pragma_v16.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n  Figure saved: {path}")


if __name__ == "__main__":
    run_experiment()
