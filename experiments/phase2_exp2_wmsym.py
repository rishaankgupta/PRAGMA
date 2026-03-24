import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

"""
PRAGMA Phase 2 — Experiment 2.2: WM+Sym (World Model + Symbolic Rules)
========================================================================
HYPOTHESIS: Injecting symbolic rules into the world model's planning
phase fixes the color algebra failure (color_swap=0%, color_chain=0%)
while preserving its spatial dynamics strengths (mirror=100%, gravity=100%).

ARCHITECTURE:
  - World model handles spatial/structural transformations (its strength)
  - Symbolic layer handles color algebra (its strength)
  - Integration point: BEFORE prediction, check if symbolic is confident;
    if yes, hand off to symbolic. If not, let world model plan.
  - This is "Pillar 2 injected into Pillar 4" — the first real integration.

HOW TO RUN:
  python3 phase2_exp2_wmsym.py
"""

import sys, os, json, time
import os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pragma.benchmark.generator import generate_benchmark
from pragma.benchmark.evaluator import evaluate_baseline
from pragma.baselines.symbolic import SymbolicBaseline, RULES
from pragma.baselines.world_model import WorldModelBaseline


class WMSymBaseline:
    """
    World Model + Symbolic integration.

    Key insight from Phase 1 data:
      - World Model fails on COLOR tasks (color_swap=0%, color_chain=0%)
        because PCA/Ridge regression has no color algebra
      - Symbolic nails COLOR tasks (100%) but struggles with spatial-count tasks
      - Spatial tasks: World Model wins (mirror=100%, gravity=100%)

    Integration: symbolic acts as a "specialist module" for color tasks.
    When symbolic is confident about a color rule, override the world model.
    When symbolic is uncertain, let world model plan spatially.
    """

    def __init__(self, sym_handoff_threshold=0.85):
        self.threshold = sym_handoff_threshold
        self._sym = SymbolicBaseline(confidence_threshold=sym_handoff_threshold)
        self._wm  = WorldModelBaseline(latent_dim=32)

        # Color-domain rules (symbolic is expert at these)
        self._color_rules = {'color_swap', 'color_chain', 'count_encode'}
        # Spatial-domain rules (world model is expert at these)
        self._spatial_rules = {'fliplr', 'flipud', 'rot90cw', 'rot90ccw',
                               'gravity', 'vertical_symmetry', 'border_fill',
                               'diagonal', 'identity'}

    def solve(self, task):
        # Step 1: Ask symbolic for its best guess + confidence
        sym_pred, explanation = self._sym.solve_with_explanation(task)
        sym_conf = explanation['confidence']
        sym_rule = explanation['selected_rule']

        # Step 2: If symbolic is confident AND rule is in color domain → use it
        if sym_conf >= self.threshold and sym_rule in self._color_rules:
            return sym_pred, 'symbolic_color', sym_conf, sym_rule

        # Step 3: If symbolic is confident about spatial rule → still use symbolic
        # (it's already good at these, no point asking WM)
        if sym_conf >= self.threshold and sym_rule in self._spatial_rules:
            return sym_pred, 'symbolic_spatial', sym_conf, sym_rule

        # Step 4: Symbolic uncertain → let world model plan
        wm_pred = self._wm.solve(task)
        return wm_pred, 'world_model', sym_conf, sym_rule

    def solve_batch(self, tasks):
        preds = []
        routing = {'symbolic_color': 0, 'symbolic_spatial': 0, 'world_model': 0}
        for task in tasks:
            pred, route, conf, rule = self.solve(task)
            preds.append(pred)
            routing[route] = routing.get(route, 0) + 1
        return preds, routing


def run_experiment():
    print("=" * 70)
    print("PRAGMA PHASE 2 — EXPERIMENT 2.2: WM+Sym (World Model + Symbolic)")
    print("=" * 70)
    print()

    tasks = generate_benchmark(n_tasks_per_type=10, seed=42)
    task_types = sorted(set(t.name for t in tasks))
    print(f"▶ Loaded {len(tasks)} tasks")
    print()

    # Reference baselines
    sym = SymbolicBaseline()
    wm  = WorldModelBaseline()
    sym_preds = sym.solve_batch(tasks)
    wm_preds  = wm.solve_batch(tasks)
    sym_res   = evaluate_baseline(tasks, sym_preds)
    wm_res    = evaluate_baseline(tasks, wm_preds)

    print(f"  Symbolic (reference):    EM={sym_res['exact_match']:.3f}")
    print(f"  World Model (reference): EM={wm_res['exact_match']:.3f}")
    print()

    # WM+Sym
    wmsym = WMSymBaseline(sym_handoff_threshold=0.85)
    t0 = time.time()
    wmsym_preds, routing = wmsym.solve_batch(tasks)
    elapsed = time.time() - t0
    wmsym_res = evaluate_baseline(tasks, wmsym_preds)
    wmsym_res['routing'] = routing

    print(f"▶ WM+Sym:  EM={wmsym_res['exact_match']:.3f}  time={elapsed:.3f}s")
    print(f"  Routing: {routing}")
    print()

    # Per-task-type
    print(f"  {'Task Type':<22} {'Symbolic':>9} {'WM':>7} {'WM+Sym':>8} {'Delta vs WM':>12}")
    print("  " + "─" * 65)
    for tt in task_types:
        s  = sym_res['by_task_type'].get(tt, 0.0)
        w  = wm_res['by_task_type'].get(tt, 0.0)
        ws = wmsym_res['by_task_type'].get(tt, 0.0)
        delta_wm = ws - w
        flag = "↑ FIXED" if delta_wm > 0.4 else ("↓ broke" if delta_wm < -0.1 else "")
        print(f"  {tt:<22} {s:>9.2f} {w:>7.2f} {ws:>8.2f} {delta_wm:>+11.2f}  {flag}")

    print()
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    best_single = max(sym_res['exact_match'], wm_res['exact_match'])
    improvement = wmsym_res['exact_match'] - best_single
    print(f"  Symbolic:    {sym_res['exact_match']:.3f}")
    print(f"  World Model: {wm_res['exact_match']:.3f}")
    print(f"  WM+Sym:      {wmsym_res['exact_match']:.3f}  ({improvement:+.3f} vs best single)")
    print()
    if improvement > 0:
        print("  ✓ Symbolic specialist injection improves world model!")
    else:
        print("  → No improvement. Interface design needs revision.")

    print()
    print("SEND THIS ENTIRE OUTPUT BACK.")
    print("=" * 70)

    # Save + figure
    os.makedirs(RESULTS_DIR + '/', exist_ok=True)
    with open(RESULTS_DIR + '//phase2_exp2_wmsym.json', 'w') as f:
        json.dump({'symbolic': sym_res, 'world_model': wm_res, 'wmsym': wmsym_res}, f, indent=2, default=str)

    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(task_types))
    w = 0.28
    ax.bar(x-w, [sym_res['by_task_type'].get(t,0) for t in task_types], w, label='Symbolic', color='#4CAF50', alpha=0.85)
    ax.bar(x,   [wm_res['by_task_type'].get(t,0) for t in task_types],  w, label='World Model', color='#9C27B0', alpha=0.85)
    ax.bar(x+w, [wmsym_res['by_task_type'].get(t,0) for t in task_types], w, label='WM+Sym', color='#FF5722', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels([t.replace('_','\n') for t in task_types], fontsize=8)
    ax.set_ylim(0,1.2); ax.set_ylabel("Exact Match"); ax.legend()
    ax.set_title("Phase 2 Exp 2.2: WM+Sym vs Individual Paradigms", fontweight='bold')
    ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR + '//phase2_exp2_wmsym.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

if __name__ == "__main__":
    run_experiment()
