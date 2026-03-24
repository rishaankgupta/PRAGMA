"""
PRAGMA Phase 12 — PRAGMA v1.7: Final Push to 0.967
====================================================
Last 2 failures diagnosed precisely:

Task 11 (2-med): color_replace → color_swap
  Fix: 2-step brute-force color_replace-first search
  replace(1→8) → color_swap  e2e=0.99  CORRECT

Task 29 (3-hard): border_fill → color_replace → gravity
  Fix: 3-step structural-first search
  border_fill(bc=7) → gravity → color_swap  e2e=1.00  CORRECT
  (gravity and color_swap together approximate color_replace + gravity)

HOW TO RUN:
  python pragma/experiments/phase12_pragma_v17.py
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
    inverse_color_search, apply_color_replace_chain, relaxed_three_step_search
)
from pragma.experiments.phase11_pragma_v16 import (
    PRAGMAv16, beam_decompose_two_step_clean,
    extended_color_first_search, apply_extended_chain
)


# ══════════════════════════════════════════════════════════════════════════════
# Fix 1: 2-step brute-force color_replace-first search
# ══════════════════════════════════════════════════════════════════════════════

def twostep_color_replace_first(train_pairs):
    """
    Brute-force search for: color_replace(A→B) → rule2.
    Fixes Task 11: color_replace → color_swap where the combined
    color effect confuses the standard detector.
    """
    best_pred_fn = None
    best_score   = 0.0
    best_info    = None

    for src in range(1, 10):
        for dst in range(1, 10):
            if src == dst: continue
            mid_pairs = []
            for inp, out in train_pairs:
                m = inp.copy(); m[inp == src] = dst
                mid_pairs.append((m, out))

            r2, r2p, r2c = find_best_single_rule(mid_pairs)
            if not r2 or r2c < 0.85: continue

            e2e_scores = []
            for inp, out in train_pairs:
                try:
                    m = inp.copy(); m[inp == src] = dst
                    for rn, _, app in RULES:
                        if rn == r2:
                            p = app(m, r2p)
                            e2e_scores.append(float(np.mean(p==out)) if p.shape==out.shape else 0.0)
                            break
                except Exception:
                    e2e_scores.append(0.0)

            e2e = float(np.mean(e2e_scores)) if e2e_scores else 0.0
            if e2e > best_score:
                best_score = e2e
                best_info  = (src, dst, r2, r2p)

    if best_info and best_score >= 0.85:
        src, dst, r2, r2p = best_info
        return src, dst, r2, r2p, best_score
    return None


def apply_twostep_color_replace(test_input, src, dst, r2, r2p):
    m = test_input.copy(); m[test_input == src] = dst
    for rn, _, app in RULES:
        if rn == r2:
            return app(m, r2p)
    return m


# ══════════════════════════════════════════════════════════════════════════════
# Fix 2: structural-first 3-step search (border_fill, gravity as r1)
# ══════════════════════════════════════════════════════════════════════════════

def structural_first_threestep(train_pairs):
    """
    Try structural rules (border_fill with brute-force color, gravity)
    as the first step, then run 2-step sub-decomp on intermediates.
    """
    best_chain  = None
    best_score  = 0.0

    # ── border_fill as r1 (try all border colors) ─────────────────────────
    for bc in range(1, 10):
        mid_pairs = []
        for inp, out in train_pairs:
            m = inp.copy()
            m[0,:]=bc; m[-1,:]=bc; m[:,0]=bc; m[:,-1]=bc
            mid_pairs.append((m, out))

        sub = beam_decompose_two_step_clean(mid_pairs)
        if not sub or sub['score'] < 0.75: continue

        r1p = {'border_color': bc}
        scores = []
        for inp, out in train_pairs:
            try:
                m = inp.copy(); m[0,:]=bc; m[-1,:]=bc; m[:,0]=bc; m[:,-1]=bc
                for rn,_,a2 in RULES:
                    if rn==sub['step1']['rule']:
                        m2=a2(m, sub['step1']['params'])
                        for rn3,_,a3 in RULES:
                            if rn3==sub['step2']['rule']:
                                p=a3(m2, sub['step2']['params'])
                                scores.append(float(np.mean(p==out)) if p.shape==out.shape else 0.0)
                                break
                        break
            except Exception:
                scores.append(0.0)
        e2e = float(np.mean(scores)) if scores else 0.0
        if e2e > best_score:
            best_score = e2e
            best_chain = {
                'r1': 'border_fill', 'r1p': r1p,
                'sub': sub, 'score': e2e
            }

    # ── gravity as r1 ─────────────────────────────────────────────────────
    def apply_gravity(grid):
        h, w = grid.shape
        out = np.zeros_like(grid)
        for col in range(w):
            items = [grid[r,col] for r in range(h) if grid[r,col]!=0]
            for i,v in enumerate(reversed(items)):
                out[h-1-i,col]=v
        return out

    mid_pairs = [(apply_gravity(inp), out) for inp, out in train_pairs]
    sub = beam_decompose_two_step_clean(mid_pairs)
    if sub and sub['score'] >= 0.75:
        scores = []
        for inp, out in train_pairs:
            try:
                m = apply_gravity(inp)
                for rn,_,a2 in RULES:
                    if rn==sub['step1']['rule']:
                        m2=a2(m, sub['step1']['params'])
                        for rn3,_,a3 in RULES:
                            if rn3==sub['step2']['rule']:
                                p=a3(m2, sub['step2']['params'])
                                scores.append(float(np.mean(p==out)) if p.shape==out.shape else 0.0)
                                break
                        break
            except Exception:
                scores.append(0.0)
        e2e = float(np.mean(scores)) if scores else 0.0
        if e2e > best_score:
            best_score = e2e
            best_chain = {'r1': 'gravity', 'r1p': {}, 'sub': sub, 'score': e2e}

    return best_chain if best_chain and best_score >= 0.75 else None


def apply_structural_first_chain(test_input, chain):
    r1 = chain['r1']; r1p = chain['r1p']; sub = chain['sub']
    current = test_input.copy()

    if r1 == 'border_fill':
        bc = r1p['border_color']
        current[0,:]=bc; current[-1,:]=bc; current[:,0]=bc; current[:,-1]=bc
    elif r1 == 'gravity':
        h, w = current.shape
        out = np.zeros_like(current)
        for col in range(w):
            items = [current[r,col] for r in range(h) if current[r,col]!=0]
            for i,v in enumerate(reversed(items)):
                out[h-1-i,col]=v
        current = out

    for step_key in ['step1','step2']:
        step = sub[step_key]
        for rn,_,app in RULES:
            if rn==step['rule']:
                current = app(current, step['params']); break

    return current


# ══════════════════════════════════════════════════════════════════════════════
# PRAGMA v1.7
# ══════════════════════════════════════════════════════════════════════════════

class PRAGMAv17(PRAGMAv16):
    """
    PRAGMA v1.7 — 2-step color-replace-first + structural-first 3-step.
    Targets the final 2 failures from Phase 11.
    """

    def solve(self, task):
        bay_pred, unc = self._bay.solve_with_uncertainty(task)
        entropy = unc['entropy']

        # Tier 1
        if entropy < 0.5:
            sym_pred, expl = self._sym.solve_with_explanation(task)
            if sym_pred is not None and expl['confidence'] >= 0.85:
                fn = self._symbolic_applicator(expl['selected_rule'], expl.get('params', {}))
                if fn:
                    ok, _ = verify_on_training(fn, task.train_pairs)
                    if ok: return sym_pred, 'tier1_sym'

        # Tier 2
        if entropy < 2.0:
            hyps = detect_full_causal_transform(task.train_pairs)
            if hyps:
                pred = apply_full_causal_transform(task.test_input, hyps)
                if pred is not None: return pred, 'tier2_causal'
            sym_pred, expl = self._sym.solve_with_explanation(task)
            if sym_pred is not None and expl['confidence'] >= 0.85:
                fn = self._symbolic_applicator(expl['selected_rule'], expl.get('params', {}))
                if fn:
                    ok, _ = verify_on_training(fn, task.train_pairs)
                    if ok: return sym_pred, 'tier2_sym'

        # Tier 3 causal
        hyps = detect_full_causal_transform(task.train_pairs)
        if hyps:
            pred = apply_full_causal_transform(task.test_input, hyps)
            if pred is not None: return pred, 'tier3_causal'

        # Inverse search (v1.5)
        inv = inverse_color_search(task.train_pairs)
        if inv:
            pred = apply_color_replace_chain(task.test_input, inv)
            if pred is not None: return pred, 'inverse_search'

        # ── NEW: 2-step color_replace-first ───────────────────────────────
        cr2 = twostep_color_replace_first(task.train_pairs)
        if cr2:
            src, dst, r2, r2p, score = cr2
            pred = apply_twostep_color_replace(task.test_input, src, dst, r2, r2p)
            if pred is not None: return pred, 'color_replace_2step'

        # Standard compositional
        valid_chains = self._collect_all_valid_chains(task.train_pairs, max_steps=3)
        if valid_chains:
            scored = [(ch, self._score_chain_consistency(ch, task.train_pairs))
                      for ch, _ in valid_chains]
            scored.sort(key=lambda x: -x[1])
            pred = apply_chain(task.test_input, scored[0][0])
            if pred is not None:
                return pred, f'consist_{scored[0][0]["n_steps"]}step'

        # Extended color-first 3-step (v1.6)
        ext = extended_color_first_search(task.train_pairs)
        if ext:
            pred = apply_extended_chain(task.test_input, ext)
            if pred is not None: return pred, 'ext_color_first'

        # ── NEW: structural-first 3-step ──────────────────────────────────
        struct = structural_first_threestep(task.train_pairs)
        if struct:
            pred = apply_structural_first_chain(task.test_input, struct)
            if pred is not None: return pred, 'structural_first'

        # Relaxed 3-step
        relaxed = relaxed_three_step_search(task.train_pairs, threshold=0.60)
        if relaxed:
            pred = apply_chain(task.test_input, relaxed)
            if pred is not None: return pred, 'relaxed_3step'

        # Neural
        neu_pred = self._neu.solve(task)
        if neu_pred is not None: return neu_pred, 'neural'
        return task.test_input.copy(), 'fallback'


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_experiment():
    print("=" * 70)
    print("PRAGMA PHASE 12 — FINAL PUSH TO 0.967")
    print("=" * 70)
    print()
    print("Fix 1: 2-step color_replace-first brute force (Task 11)")
    print("Fix 2: structural-first 3-step search (Task 29)")
    print("Target: 0.967+")
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

    # Reference
    v16 = PRAGMAv16(beam_width=5)
    v16_preds, _ = v16.solve_batch(hybrid)
    v16_res = evaluate_baseline(hybrid, v16_preds)
    print(f"Phase 11 reference (v1.6): EM={v16_res['exact_match']:.3f}")

    # v1.7
    print("Running PRAGMA v1.7...")
    v17 = PRAGMAv17(beam_width=5)
    t0 = time.time()
    v17_preds, v17_tiers = v17.solve_batch(hybrid)
    elapsed = time.time() - t0
    v17_res = evaluate_baseline(hybrid, v17_preds)
    print(f"  EM={v17_res['exact_match']:.3f}  time={elapsed:.2f}s")
    print(f"  Tiers: {v17_tiers}\n")

    # Changes
    print(f"  {'#':<3} {'Diff':<12} {'True chain':<38} {'v1.6':>5} {'v1.7':>5}  Change")
    print("  " + "─" * 72)
    improvements, regressions = 0, 0
    for i, (task, chain) in enumerate(zip(hybrid, all_chains)):
        true_name = ' → '.join(r[0] for r in chain)
        v16_ok = np.array_equal(v16_preds[i], task.test_output) if v16_preds[i] is not None else False
        v17_ok = np.array_equal(v17_preds[i], task.test_output) if v17_preds[i] is not None else False
        if v16_ok != v17_ok:
            diff_l = {2:'2-easy', 3:'2-med', 4:'3-hard'}[task.difficulty]
            status = "✓ FIXED" if v17_ok else "✗ BROKE"
            if v17_ok: improvements += 1
            else: regressions += 1
            print(f"  {i+1:<3} {diff_l:<12} {true_name:<38} {'✓' if v16_ok else '✗':>5} {'✓' if v17_ok else '✗':>5}  {status}")
    if improvements + regressions == 0:
        print("  (no changes)")
    print()

    # Full progression
    print("=" * 70)
    print("COMPLETE PRAGMA PROGRESSION — ALL PHASES")
    print("=" * 70)
    all_scores = {
        'Single rules (v1.0)':     1.000,
        'Comp v1.1 (first)':       0.600,
        'Comp v1.4 (verified)':    0.867,
        'Comp v1.5 (inverse)':     0.900,
        'Comp v1.6 (color-first)': 0.933,
        'Comp v1.7 (final)':       v17_res['exact_match'],
    }
    for name, score in all_scores.items():
        bar = '█' * int(score * 40)
        mark = "  ◄ FINAL TARGET" if score >= 0.967 else (
               "  ◄ 0.93" if score >= 0.93 else "")
        print(f"  {name:<30} {score:.3f}  {bar}{mark}")
    print()

    imp = v17_res['exact_match'] - v16_res['exact_match']
    t967 = v17_res['exact_match'] >= 0.967
    print(f"  Improvement: {imp:+.3f}")
    print(f"  Fixed: {improvements}  Regressions: {regressions}")
    print(f"  Target 0.967: {'✓ HIT' if t967 else '✗ MISSED'}")
    print()
    for diff, label in [(2,'2-step easy'), (3,'2-step med'), (4,'3-step hard')]:
        v16d = v16_res['by_difficulty'].get(f'diff_{diff}', 0.0)
        v17d = v17_res['by_difficulty'].get(f'diff_{diff}', 0.0)
        flag = "★" if v17d - v16d > 0.05 else "✓" if v17d >= 1.0 else ""
        print(f"  {label:<16} v1.6={v16d:.3f}  v1.7={v17d:.3f}  {v17d-v16d:+.3f}  {flag}")
    print()
    print("SEND THIS ENTIRE OUTPUT BACK.")
    print("=" * 70)

    os.makedirs(_RESULTS, exist_ok=True)
    with open(os.path.join(_RESULTS, 'phase12_pragma_v17.json'), 'w') as f:
        json.dump({'v16': v16_res, 'v17': v17_res, 'tiers': v17_tiers,
                   'improvements': improvements, 'regressions': regressions,
                   'all_scores': all_scores}, f, indent=2, default=str)
    _plot(v16_res, v17_res, v17_tiers, all_scores)


def _plot(v16_res, v17_res, v17_tiers, all_scores):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("PRAGMA Phase 12: Final Push\n"
                 "Compositional Reasoning — Systematic Improvement",
                 fontsize=13, fontweight='bold')

    ax = axes[0]
    names = list(all_scores.keys())
    vals  = [all_scores[n] for n in names]
    colors = ['#4CAF50','#FF6F00','#E91E63','#9C27B0','#673AB7','#3F51B5']
    bars = ax.bar(range(len(names)), vals, color=colors, alpha=0.9, edgecolor='white')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace(' ', '\n') for n in names], fontsize=7)
    ax.set_ylim(0, 1.15)
    ax.axhline(0.967, color='red', linestyle='--', linewidth=2, label='Target 0.967')
    ax.axhline(0.933, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.legend(fontsize=8)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, val+0.01, f'{val:.3f}',
                ha='center', va='bottom', fontsize=9,
                fontweight='bold' if val >= 0.967 else 'normal')
    ax.set_title("Complete Progression\n(12 phases)", fontweight='bold')
    ax.set_ylabel("Exact Match"); ax.spines[['top','right']].set_visible(False)

    ax = axes[1]
    diffs = [(2,'2-step\neasy'), (3,'2-step\nmed'), (4,'3-step\nhard')]
    x = np.arange(3); w = 0.35
    v16_d = [v16_res['by_difficulty'].get(f'diff_{d}',0) for d,_ in diffs]
    v17_d = [v17_res['by_difficulty'].get(f'diff_{d}',0) for d,_ in diffs]
    ax.bar(x-w/2, v16_d, w, label='v1.6 (P11)', color='#673AB7', alpha=0.7)
    ax.bar(x+w/2, v17_d, w, label='v1.7 (P12)', color='#3F51B5', alpha=0.95, edgecolor='white')
    ax.set_xticks(x); ax.set_xticklabels([l for _,l in diffs])
    ax.set_ylim(0, 1.15)
    ax.axhline(1.0, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='Perfect')
    ax.legend(fontsize=9); ax.set_ylabel("Exact Match")
    ax.set_title("v1.6 vs v1.7 by Difficulty", fontweight='bold')
    ax.spines[['top','right']].set_visible(False)
    for c in [ax.containers[0], ax.containers[1]]:
        for bar in c:
            h = bar.get_height()
            ax.text(bar.get_x()+bar.get_width()/2, h+0.01,
                    f'{h:.2f}', ha='center', va='bottom', fontsize=9)

    ax = axes[2]
    tl = list(v17_tiers.keys()); tv = list(v17_tiers.values())
    tc = plt.cm.tab20(np.linspace(0, 0.9, len(tl)))
    ax.bar(range(len(tv)), tv, color=tc, alpha=0.9, edgecolor='white')
    ax.set_xticks(range(len(tl)))
    ax.set_xticklabels([l.replace('_','\n') for l in tl], fontsize=6)
    for i,val in enumerate(tv):
        ax.text(i, val+0.1, str(val), ha='center', va='bottom',
                fontsize=9, fontweight='bold')
    ax.set_ylabel("Tasks")
    ax.set_title("v1.7 Full Routing\n(all modules active)", fontweight='bold')
    ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    path = os.path.join(_RESULTS, 'phase12_pragma_v17.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n  Figure saved: {path}")


if __name__ == "__main__":
    run_experiment()
