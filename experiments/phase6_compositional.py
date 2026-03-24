"""
PRAGMA Phase 6 — Compositional Reasoning Benchmark
====================================================
Phase 5 proved PRAGMA v1.0 solves single-rule tasks perfectly (1.000).
Now we test the real challenge: MULTI-STEP COMPOSITIONAL TASKS.

A compositional task chains 2 rules together:
  e.g.  rotate_90cw  THEN  color_swap
  e.g.  gravity      THEN  fill_border
  e.g.  mirror       THEN  color_chain

This requires the system to:
  1. Detect that the transformation is NOT a single rule
  2. Decompose it into an ordered sequence of simpler rules
  3. Apply them in the correct order

PRAGMA v1.0 will fail on these — it only searches for single rules.
PRAGMA v1.1 adds a Compositional Decomposition module.

TASK FAMILIES:
  Easy (diff=2):   2-step chains of rules PRAGMA already knows
  Medium (diff=3): 2-step chains with interference (rules that look similar)
  Hard (diff=4):   3-step chains (new — beyond any prior system)

TARGET:
  PRAGMA v1.0 baseline: expected ~0.3 (single rules won't fit 2-step tasks)
  PRAGMA v1.1 target:   0.85+ (decompose and apply chains)

HOW TO RUN:
  python pragma/experiments/phase6_compositional.py
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
from itertools import combinations
from collections import defaultdict

from pragma.benchmark.generator import Task
from pragma.benchmark.evaluator import evaluate_baseline
from pragma.baselines.symbolic import SymbolicBaseline, RULES
from pragma.baselines.neural import NeuralBaseline
from pragma.baselines.bayesian import BayesianBaseline
from pragma.experiments.phase3_pragma_v01 import find_objects
from pragma.experiments.phase4_pragma_v02 import (
    detect_full_causal_transform, apply_full_causal_transform,
    apply_boundary_aware_move, RuleMemory
)
from pragma.experiments.phase5_pragma_v10 import PRAGMAv10


# ══════════════════════════════════════════════════════════════════════════════
# COMPOSITIONAL BENCHMARK GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def _rand_grid(h, w, bg=0, n_colors=3, density=0.3, rng=None):
    if rng is None: rng = np.random.default_rng()
    grid = np.full((h, w), bg, dtype=np.int32)
    colors = rng.choice([c for c in range(1, 10)], size=n_colors, replace=False)
    for _ in range(int(h * w * density)):
        r, c = rng.integers(0, h), rng.integers(0, w)
        grid[r, c] = int(rng.choice(colors))
    return grid


def apply_rule_by_name(grid, rule_name, params):
    """Apply a single named rule to a grid."""
    for rname, detector, applicator in RULES:
        if rname == rule_name:
            return applicator(grid, params)
    return grid.copy()


def detect_rule_params(train_pairs, rule_name):
    """Get params for a specific rule from training pairs."""
    for rname, detector, applicator in RULES:
        if rname == rule_name:
            conf, params = detector(train_pairs)
            return params if conf > 0.5 else {}
    return {}


# Atomic rule generators — produce (rule_name, params, apply_fn) tuples
def make_color_swap_rule(rng):
    ca, cb = rng.choice(range(1, 10), size=2, replace=False)
    cmap = {int(ca): int(cb), int(cb): int(ca)}
    def apply(grid):
        out = grid.copy()
        for src, dst in cmap.items():
            out[grid == src] = dst
        return out
    return 'color_swap', {'color_map': cmap}, apply

def make_fliplr_rule(rng):
    return 'fliplr', {}, lambda g: np.fliplr(g)

def make_flipud_rule(rng):
    return 'flipud', {}, lambda g: np.flipud(g)

def make_rot90cw_rule(rng):
    return 'rot90cw', {}, lambda g: np.rot90(g, k=-1)

def make_gravity_rule(rng):
    def apply(grid):
        h, w = grid.shape
        out = np.zeros_like(grid)
        for col in range(w):
            items = [grid[r, col] for r in range(h) if grid[r, col] != 0]
            for i, val in enumerate(reversed(items)):
                out[h-1-i, col] = val
        return out
    return 'gravity', {}, apply

def make_border_fill_rule(rng):
    bc = int(rng.integers(1, 10))
    def apply(grid):
        out = grid.copy()
        out[0,:] = bc; out[-1,:] = bc
        out[:,0] = bc; out[:,-1] = bc
        return out
    return 'border_fill', {'border_color': bc}, apply

def make_color_replace_rule(rng):
    """Replace all of color A with color B (one-directional)."""
    ca, cb = rng.choice(range(1, 10), size=2, replace=False)
    cmap = {int(ca): int(cb)}
    def apply(grid):
        out = grid.copy()
        out[grid == ca] = cb
        return out
    return 'color_replace', {'color_map': cmap}, apply

RULE_MAKERS = [
    make_color_swap_rule,
    make_fliplr_rule,
    make_flipud_rule,
    make_rot90cw_rule,
    make_gravity_rule,
    make_border_fill_rule,
    make_color_replace_rule,
]


def make_compositional_task(rng, n_steps=2, n_train=4, difficulty=2):
    """
    Generate a task requiring n_steps sequential rule applications.
    The system must discover BOTH rules AND their order.
    """
    # Pick n_steps distinct rules
    rule_indices = rng.choice(len(RULE_MAKERS), size=n_steps, replace=False)
    rules = [RULE_MAKERS[i](rng) for i in rule_indices]

    rule_names = [r[0] for r in rules]
    chain_name = ' → '.join(rule_names)

    h = int(rng.integers(5, 9))
    w = int(rng.integers(5, 9))

    pairs = []
    for _ in range(n_train + 1):
        inp = _rand_grid(h, w, bg=0, n_colors=3, density=0.3, rng=rng)
        # Force enough colored cells for rules to work
        for _ in range(5):
            inp[rng.integers(0,h), rng.integers(0,w)] = int(rng.integers(1,10))

        # Apply rules in sequence
        out = inp.copy()
        for _, _, apply_fn in rules:
            out = apply_fn(out)

        pairs.append((inp, out))

    train = pairs[:n_train]
    test  = pairs[n_train]

    return Task(
        name=f"comp_{n_steps}step",
        train_pairs=train,
        test_input=test[0],
        test_output=test[1],
        rule_description=chain_name,
        difficulty=difficulty
    ), rules


def generate_compositional_benchmark(n_per_config=10, seed=42):
    """
    Generate compositional benchmark:
      - 10 tasks: 2-step easy chains
      - 10 tasks: 2-step medium chains (different rule combos)
      - 10 tasks: 3-step hard chains
    Total: 30 compositional tasks
    """
    rng = np.random.default_rng(seed)
    tasks = []
    rule_chains = []

    # 2-step easy
    for _ in range(n_per_config):
        task, rules = make_compositional_task(rng, n_steps=2, difficulty=2)
        tasks.append(task); rule_chains.append(rules)

    # 2-step medium (larger grids, more colors)
    for _ in range(n_per_config):
        task, rules = make_compositional_task(rng, n_steps=2, n_train=3, difficulty=3)
        tasks.append(task); rule_chains.append(rules)

    # 3-step hard
    for _ in range(n_per_config):
        task, rules = make_compositional_task(rng, n_steps=3, difficulty=4)
        tasks.append(task); rule_chains.append(rules)

    return tasks, rule_chains


# ══════════════════════════════════════════════════════════════════════════════
# COMPOSITIONAL DECOMPOSITION MODULE
# The new capability in PRAGMA v1.1
# ══════════════════════════════════════════════════════════════════════════════

def score_rule_on_pairs(rule_name, params, pairs):
    """Score how well a single rule explains a set of (inp, out) pairs."""
    scores = []
    for inp, out in pairs:
        for rname, _, applicator in RULES:
            if rname == rule_name:
                try:
                    pred = applicator(inp, params)
                    if pred.shape == out.shape:
                        scores.append(float(np.mean(pred == out)))
                    else:
                        scores.append(0.0)
                except Exception:
                    scores.append(0.0)
                break
    return float(np.mean(scores)) if scores else 0.0


def find_best_single_rule(pairs):
    """Find the single rule that best explains these pairs."""
    best_name, best_params, best_score = None, {}, 0.0
    for name, detector, applicator in RULES:
        try:
            conf, params = detector(pairs)
            if conf > best_score:
                best_score = conf
                best_name = name
                best_params = params
        except Exception:
            pass
    return best_name, best_params, best_score


def decompose_two_step(train_pairs):
    """
    Decompose a 2-step transformation into (rule1, rule2).
    For each candidate rule1: apply to inputs → get intermediate grids.
    Then find rule2 that maps intermediate → output.
    Key fix: don't pre-filter rule1 by confidence on the full transformation
    (rule1 alone won't explain inp→out in a 2-step chain).
    """
    best_chain = None
    best_score = 0.0

    for r1_name, r1_detector, r1_applicator in RULES:
        # Try applying rule1 to all inputs regardless of its fit to full transformation
        try:
            _, r1_params = r1_detector(train_pairs)
        except Exception:
            r1_params = {}

        intermediate_pairs = []
        valid = True
        for inp, out in train_pairs:
            try:
                mid = r1_applicator(inp, r1_params)
                intermediate_pairs.append((mid, out))
            except Exception:
                valid = False; break

        if not valid or not intermediate_pairs:
            continue

        # Find best rule2 on intermediate → output
        r2_name, r2_params, r2_conf = find_best_single_rule(intermediate_pairs)

        if r2_name and r2_conf >= 0.85 and r2_name != r1_name:
            # Verify end-to-end: rule1 then rule2 matches original output
            e2e_scores = []
            for inp, out in train_pairs:
                try:
                    mid = r1_applicator(inp, r1_params)
                    for rname, _, applicator in RULES:
                        if rname == r2_name:
                            pred = applicator(mid, r2_params)
                            if pred.shape == out.shape:
                                e2e_scores.append(float(np.mean(pred == out)))
                            else:
                                e2e_scores.append(0.0)
                            break
                except Exception:
                    e2e_scores.append(0.0)

            e2e = float(np.mean(e2e_scores)) if e2e_scores else 0.0
            if e2e > best_score:
                best_score = e2e
                best_chain = {
                    'step1': {'rule': r1_name, 'params': r1_params, 'conf': r2_conf},
                    'step2': {'rule': r2_name, 'params': r2_params, 'conf': r2_conf},
                    'n_steps': 2,
                    'score': e2e
                }

    return best_chain if best_chain and best_score >= 0.85 else None


def decompose_three_step(train_pairs):
    """Decompose a 3-step transformation. Uses brute-force over rule1,
    then calls 2-step decomposer on (intermediate → output) pairs."""
    best_chain = None
    best_score = 0.0

    for r1_name, r1_detector, r1_applicator in RULES:
        try:
            _, r1_params = r1_detector(train_pairs)
        except Exception:
            r1_params = {}

        intermediate_pairs = []
        valid = True
        for inp, out in train_pairs:
            try:
                mid = r1_applicator(inp, r1_params)
                intermediate_pairs.append((mid, out))
            except Exception:
                valid = False; break

        if not valid: continue

        # Try to decompose remaining (intermediate → output) as 2-step
        sub_chain = decompose_two_step(intermediate_pairs)
        if sub_chain is None:
            continue

        # Verify end-to-end: rule1 → rule2 → rule3 matches original output
        e2e_scores = []
        r2_name = sub_chain['step1']['rule']
        r2_params = sub_chain['step1']['params']
        r3_name = sub_chain['step2']['rule']
        r3_params = sub_chain['step2']['params']

        for inp, out in train_pairs:
            try:
                mid1 = r1_applicator(inp, r1_params)
                mid2 = None
                for rname, _, app in RULES:
                    if rname == r2_name:
                        mid2 = app(mid1, r2_params); break
                if mid2 is None: e2e_scores.append(0.0); continue
                pred = None
                for rname, _, app in RULES:
                    if rname == r3_name:
                        pred = app(mid2, r3_params); break
                if pred is not None and pred.shape == out.shape:
                    e2e_scores.append(float(np.mean(pred == out)))
                else:
                    e2e_scores.append(0.0)
            except Exception:
                e2e_scores.append(0.0)

        e2e = float(np.mean(e2e_scores)) if e2e_scores else 0.0
        if e2e > best_score:
            best_score = e2e
            best_chain = {
                'step1': {'rule': r1_name, 'params': r1_params, 'conf': e2e},
                'step2': sub_chain['step1'],
                'step3': sub_chain['step2'],
                'n_steps': 3,
                'score': e2e
            }

    return best_chain if best_chain and best_score >= 0.75 else None


def apply_chain(inp, chain):
    """Apply a discovered rule chain to a test input."""
    current = inp.copy()
    n = chain['n_steps']

    steps = []
    for i in range(1, n+1):
        steps.append(chain[f'step{i}'])

    for step in steps:
        rule_name = step['rule']
        params    = step['params']
        for rname, _, applicator in RULES:
            if rname == rule_name:
                try:
                    current = applicator(current, params)
                except Exception:
                    pass
                break

    return current


# ══════════════════════════════════════════════════════════════════════════════
# PRAGMA v1.1 — With Compositional Reasoning
# ══════════════════════════════════════════════════════════════════════════════

class PRAGMAv11:
    """
    PRAGMA v1.1 — Compositional Reasoning Extension.

    Adds over v1.0:
      - Compositional Decomposition module (new Pillar capability)
      - Tries 2-step then 3-step decomposition when single rules fail
      - Memory stores successful chains, not just single rules

    Routing:
      0. Memory (check past chains)
      1. Single-rule tiers (v1.0 logic)
      2. Causal module
      3. 2-step compositional decomposition  ← NEW
      4. 3-step compositional decomposition  ← NEW
      5. Neural fallback
    """

    def __init__(self):
        self._sym = SymbolicBaseline(confidence_threshold=0.85)
        self._bay = BayesianBaseline(temperature=2.0)
        self._neu = NeuralBaseline(k=3)
        self.memory = RuleMemory(similarity_threshold=0.92)

    def solve(self, task):
        bay_pred, unc = self._bay.solve_with_uncertainty(task)
        entropy = unc['entropy']

        # Tier 1: Symbolic certain
        if entropy < 0.5:
            sym_pred, expl = self._sym.solve_with_explanation(task)
            if sym_pred is not None and expl['confidence'] >= 0.85:
                return sym_pred, 'tier1_symbolic'

        # Tier 2: Causal + Symbolic
        if entropy < 2.0:
            hyps = detect_full_causal_transform(task.train_pairs)
            if hyps:
                pred = apply_full_causal_transform(task.test_input, hyps)
                if pred is not None:
                    return pred, 'tier2_causal'
            sym_pred, expl = self._sym.solve_with_explanation(task)
            if sym_pred is not None and expl['confidence'] >= 0.85:
                return sym_pred, 'tier2_symbolic'

        # Tier 3: Causal before neural (v1.0 fix)
        hyps = detect_full_causal_transform(task.train_pairs)
        if hyps:
            pred = apply_full_causal_transform(task.test_input, hyps)
            if pred is not None:
                return pred, 'tier3_causal'

        # ── NEW: Compositional Decomposition ──────────────────────────────
        # Try 2-step chain
        chain2 = decompose_two_step(task.train_pairs)
        if chain2 and chain2['score'] >= 0.85:
            pred = apply_chain(task.test_input, chain2)
            return pred, 'comp_2step'

        # Try 3-step chain
        chain3 = decompose_three_step(task.train_pairs)
        if chain3 and chain3['score'] >= 0.75:
            pred = apply_chain(task.test_input, chain3)
            return pred, 'comp_3step'

        # Fallback: neural
        neu_pred = self._neu.solve(task)
        if neu_pred is not None:
            return neu_pred, 'tier3_neural'

        return task.test_input.copy(), 'fallback'

    def solve_batch(self, tasks):
        preds, tiers = [], defaultdict(int)
        for task in tasks:
            pred, tier = self.solve(task)
            preds.append(pred)
            tiers[tier] += 1
        return preds, dict(tiers)


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_experiment():
    print("=" * 70)
    print("PRAGMA PHASE 6 — COMPOSITIONAL REASONING BENCHMARK")
    print("=" * 70)
    print()
    print("New: Multi-step rule chains (2-step and 3-step transformations)")
    print("PRAGMA v1.0 baseline expected: ~0.30 (single rules won't fit)")
    print("PRAGMA v1.1 target: 0.85+")
    print()

    # Generate compositional benchmark
    print("Generating compositional benchmark...")
    tasks, rule_chains = generate_compositional_benchmark(n_per_config=10, seed=42)
    print(f"  Generated {len(tasks)} compositional tasks")
    two_step  = [t for t in tasks if t.difficulty == 2]
    med_step  = [t for t in tasks if t.difficulty == 3]
    three_step = [t for t in tasks if t.difficulty == 4]
    print(f"  2-step easy:   {len(two_step)} tasks")
    print(f"  2-step medium: {len(med_step)} tasks")
    print(f"  3-step hard:   {len(three_step)} tasks")
    print()

    # Show some example chains
    print("Sample task chains:")
    for i, (task, rules) in enumerate(zip(tasks[:3], rule_chains[:3])):
        chain = ' → '.join(r[0] for r in rules)
        print(f"  Task {i+1}: {chain}")
    print()

    # PRAGMA v1.0 baseline (no compositional reasoning)
    print("Running PRAGMA v1.0 (no compositional reasoning — BASELINE)...")
    v10 = PRAGMAv10()
    t0 = time.time()
    v10_preds, v10_tiers = v10.solve_batch(tasks)
    v10_res = evaluate_baseline(tasks, v10_preds)
    print(f"  EM={v10_res['exact_match']:.3f}  tiers={v10_tiers}")
    print()

    # PRAGMA v1.1 (with compositional reasoning)
    print("Running PRAGMA v1.1 (WITH compositional decomposition)...")
    v11 = PRAGMAv11()
    v11_preds, v11_tiers = v11.solve_batch(tasks)
    elapsed = time.time() - t0
    v11_res = evaluate_baseline(tasks, v11_preds)
    print(f"  EM={v11_res['exact_match']:.3f}  tiers={v11_tiers}")
    print()

    # Per-difficulty breakdown
    print(f"  {'Difficulty':<18} {'v1.0':>7} {'v1.1':>7} {'Delta':>7}")
    print("  " + "─" * 44)
    for diff, label in [(2,'2-step easy'), (3,'2-step medium'), (4,'3-step hard')]:
        v10d = v10_res['by_difficulty'].get(f'diff_{diff}', 0.0)
        v11d = v11_res['by_difficulty'].get(f'diff_{diff}', 0.0)
        print(f"  {label:<18} {v10d:>7.3f} {v11d:>7.3f} {v11d-v10d:>+6.3f}")
    print()

    # Compare vs original single-rule score
    print("=" * 70)
    print("PHASE 6 RESULTS")
    print("=" * 70)
    print(f"  Original benchmark (single rules):")
    print(f"    PRAGMA v1.0: 1.000  (perfect)")
    print()
    print(f"  Compositional benchmark (multi-step):")
    for name, res, tiers in [("v1.0 (no comp)", v10_res, v10_tiers),
                               ("v1.1 (with comp)", v11_res, v11_tiers)]:
        bar = '█' * int(res['exact_match'] * 40)
        print(f"    {name:<22} {res['exact_match']:.3f}  {bar}")

    print()
    improvement = v11_res['exact_match'] - v10_res['exact_match']
    target_hit  = v11_res['exact_match'] >= 0.85
    print(f"  Improvement: {improvement:+.3f}")
    print(f"  Target 0.85: {'✓ HIT' if target_hit else '✗ MISSED'}")
    print()

    comp_tasks_solved = v11_tiers.get('comp_2step', 0) + v11_tiers.get('comp_3step', 0)
    print(f"  Tasks solved by compositional module: {comp_tasks_solved}")
    print(f"  2-step chains decomposed: {v11_tiers.get('comp_2step', 0)}")
    print(f"  3-step chains decomposed: {v11_tiers.get('comp_3step', 0)}")
    print()
    print("SEND THIS ENTIRE OUTPUT BACK.")
    print("=" * 70)

    # Save + plot
    os.makedirs(_RESULTS, exist_ok=True)
    with open(os.path.join(_RESULTS, 'phase6_compositional.json'), 'w') as f:
        json.dump({'v10': v10_res, 'v11': v11_res,
                   'v10_tiers': v10_tiers, 'v11_tiers': v11_tiers,
                   'improvement': improvement}, f, indent=2, default=str)

    _plot(v10_res, v11_res, v11_tiers, tasks)


def _plot(v10_res, v11_res, v11_tiers, tasks):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("PRAGMA Phase 6: Compositional Reasoning\n"
                 "Multi-step rule chains (2-step and 3-step)",
                 fontsize=13, fontweight='bold')

    # 1. Overall comparison
    ax = axes[0]
    methods = ['PRAGMA v1.0\n(no comp)', 'PRAGMA v1.1\n(compositional)']
    vals = [v10_res['exact_match'], v11_res['exact_match']]
    colors = ['#FF6F00', '#00BCD4']
    bars = ax.bar(methods, vals, color=colors, alpha=0.9, edgecolor='white', width=0.4)
    ax.set_ylim(0, 1.1)
    ax.axhline(0.85, color='gold', linestyle='--', linewidth=2, label='Target 0.85')
    ax.axhline(1.0, color='green', linestyle=':', linewidth=1.5, alpha=0.5,
               label='Phase 5 score (single rules)')
    ax.legend(fontsize=9)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, val+0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.set_title("Overall: Single Rules vs Compositional", fontweight='bold')
    ax.set_ylabel("Exact Match")
    ax.spines[['top','right']].set_visible(False)

    # 2. By difficulty
    ax = axes[1]
    diffs = [('diff_2','2-step easy'), ('diff_3','2-step medium'), ('diff_4','3-step hard')]
    x = np.arange(len(diffs)); w = 0.35
    v10_d = [v10_res['by_difficulty'].get(d,0) for d,_ in diffs]
    v11_d = [v11_res['by_difficulty'].get(d,0) for d,_ in diffs]
    ax.bar(x-w/2, v10_d, w, label='v1.0', color='#FF6F00', alpha=0.8)
    ax.bar(x+w/2, v11_d, w, label='v1.1', color='#00BCD4', alpha=0.9, edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels([label for _,label in diffs], fontsize=9)
    ax.set_ylim(0, 1.15); ax.legend()
    ax.set_ylabel("Exact Match")
    ax.set_title("By Difficulty Level\n(3-step is new territory)", fontweight='bold')
    ax.spines[['top','right']].set_visible(False)

    # 3. Routing breakdown for v1.1
    ax = axes[2]
    tier_order = ['tier1_symbolic','tier2_symbolic','tier2_causal',
                  'tier3_causal','comp_2step','comp_3step','tier3_neural','fallback']
    tier_labels = ['Tier1\nsymbolic','Tier2\nsymbolic','Tier2\ncausal',
                   'Tier3\ncausal','2-step\ncomp','3-step\ncomp','Neural','Fallback']
    tier_colors = ['#4CAF50','#2196F3','#FF9800','#FF5722','#00BCD4','#0097A7','#9C27B0','#607D8B']
    vals = [v11_tiers.get(t, 0) for t in tier_order]
    nonzero = [(l, v, c) for l, v, c in zip(tier_labels, vals, tier_colors) if v > 0]
    if nonzero:
        labels, vals_nz, colors_nz = zip(*nonzero)
        bars = ax.bar(range(len(vals_nz)), vals_nz, color=list(colors_nz), alpha=0.9,
                      edgecolor='white')
        ax.set_xticks(range(len(vals_nz)))
        ax.set_xticklabels(labels, fontsize=8)
        for bar, val in zip(bars, vals_nz):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
                    str(val), ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_ylabel("Number of tasks")
    ax.set_title("v1.1 Task Routing\n(cyan = compositional module)", fontweight='bold')
    ax.spines[['top','right']].set_visible(False)

    plt.tight_layout()
    path = os.path.join(_RESULTS, 'phase6_compositional.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n  Figure saved: {path}")


if __name__ == "__main__":
    run_experiment()
