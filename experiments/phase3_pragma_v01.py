"""
PRAGMA Phase 3 — PRAGMA v0.1: The Three-Pillar Core
=====================================================
This is the first real unified system. Not just combining outputs —
actually integrating the paradigms at the representation level.

ARCHITECTURE:
  Pillar 1 (Perception+Neural):   Feature extraction + transformation detection
  Pillar 2 (Symbolic+Causal):     Rule library + object tracking + counting
  Pillar 3 (Bayesian):            Uncertainty routing — decides who gets to speak

NEW COMPONENTS vs Phase 2:
  - Object tracker: detects discrete objects, tracks positions across pairs
  - Counting module: explicit count(color) predicate — fixes count_encode=0%
  - Causal delta detector: infers (object, position, movement_vector) tuples
  - Confidence router: 3-tier routing based on posterior entropy
    Tier 1 (entropy < 0.5):  symbolic certain → use symbolic
    Tier 2 (0.5-1.5):        ambiguous → run object tracker + causal module
    Tier 3 (entropy > 1.5):  no rule found → neural generalization

TARGET: 0.95+ exact match (solve object_move AND count_encode)

HOW TO RUN:
  python pragma/experiments/phase3_pragma_v01.py
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

from pragma.benchmark.generator import generate_benchmark
from pragma.benchmark.evaluator import evaluate_baseline
from pragma.baselines.symbolic import SymbolicBaseline, RULES
from pragma.baselines.neural import NeuralBaseline, extract_features, apply_learned_transformation
from pragma.baselines.bayesian import BayesianBaseline


# ══════════════════════════════════════════════════════════════════════════════
# NEW MODULE 1: Object Tracker
# Detects discrete objects (connected components) and their positions.
# Enables causal reasoning: "what moved, by how much, in what direction?"
# ══════════════════════════════════════════════════════════════════════════════

def find_objects(grid):
    """
    Find discrete objects (connected components of same color) in a grid.
    Returns list of {'color': int, 'cells': [(r,c)], 'centroid': (r,c), 'bbox': (r0,c0,r1,c1)}
    """
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    objects = []

    def bfs(start_r, start_c, color):
        cells = []
        queue = [(start_r, start_c)]
        visited[start_r, start_c] = True
        while queue:
            r, c = queue.pop(0)
            cells.append((r, c))
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < h and 0 <= nc < w and not visited[nr,nc] and grid[nr,nc] == color:
                    visited[nr,nc] = True
                    queue.append((nr,nc))
        return cells

    for r in range(h):
        for c in range(w):
            if not visited[r,c] and grid[r,c] != 0:
                color = int(grid[r,c])
                cells = bfs(r, c, color)
                rs = [cell[0] for cell in cells]
                cs = [cell[1] for cell in cells]
                objects.append({
                    'color': color,
                    'cells': cells,
                    'centroid': (float(np.mean(rs)), float(np.mean(cs))),
                    'bbox': (min(rs), min(cs), max(rs), max(cs)),
                    'size': len(cells)
                })
    return objects


def detect_object_movement(inp, out):
    """
    Detect if a single object moved between input and output.
    Returns (delta_r, delta_c, color, confidence) or None.
    """
    obj_in  = find_objects(inp)
    obj_out = find_objects(out)

    # Only works cleanly with 1 object
    if len(obj_in) != 1 or len(obj_out) != 1:
        return None
    if obj_in[0]['color'] != obj_out[0]['color']:
        return None
    if obj_in[0]['size'] != obj_out[0]['size']:
        return None

    cr_in,  cc_in  = obj_in[0]['centroid']
    cr_out, cc_out = obj_out[0]['centroid']
    dr = cr_out - cr_in
    dc = cc_out - cc_in

    # Verify: applying this delta to input cells gives output cells
    expected_out = np.zeros_like(inp)
    h, w = inp.shape
    color = obj_in[0]['color']
    valid = True
    for (r, c) in obj_in[0]['cells']:
        nr = int(round(r + dr))
        nc = int(round(c + dc))
        if 0 <= nr < h and 0 <= nc < w:
            expected_out[nr, nc] = color
        else:
            valid = False; break

    if not valid:
        return None

    match = float(np.mean(expected_out == out))
    return (dr, dc, color, match)


# ══════════════════════════════════════════════════════════════════════════════
# NEW MODULE 2: Counting Module
# Explicit count(color) → bar encoding predicate.
# Fixes Symbolic=0% on count_encode tasks.
# ══════════════════════════════════════════════════════════════════════════════

def detect_count_rule(train_pairs):
    """
    Detect if output encodes the count of a specific color as a bar.
    Returns (color_to_count, bar_color, confidence) or None.
    """
    best = None
    best_conf = 0.0

    for target_color in range(1, 10):
        for bar_color in range(1, 10):
            scores = []
            for inp, out in train_pairs:
                if out.shape[0] != 1:
                    scores.append(0.0); continue
                count = int(np.sum(inp == target_color))
                count = max(0, min(count, out.shape[1]))
                expected = np.zeros_like(out)
                expected[0, :count] = bar_color
                scores.append(float(np.array_equal(expected, out)))

            conf = float(np.mean(scores)) if scores else 0.0
            if conf > best_conf:
                best_conf = conf
                best = (target_color, bar_color, conf)

    if best and best[2] >= 0.8:
        return best
    return None


def apply_count_rule(inp, target_color, bar_color):
    """Apply count encoding: count target_color cells, output as bar."""
    count = int(np.sum(inp == target_color))
    w = inp.shape[1]
    count = max(0, min(count, w))
    out = np.zeros((1, w), dtype=np.int32)
    out[0, :count] = bar_color
    return out


# ══════════════════════════════════════════════════════════════════════════════
# NEW MODULE 3: Causal Delta Detector
# Builds a causal model of the transformation:
# "What properties of the input CAUSE which properties of the output?"
# ══════════════════════════════════════════════════════════════════════════════

def detect_causal_transform(train_pairs):
    """
    Detect causal transformation: what input feature predicts the output?
    Returns dict of causal hypotheses with confidence scores.
    """
    hypotheses = {}

    # Hypothesis: object movement (causal: position_in → position_out)
    move_vectors = []
    for inp, out in train_pairs:
        result = detect_object_movement(inp, out)
        if result:
            dr, dc, color, conf = result
            if conf > 0.9:
                move_vectors.append((dr, dc))

    if len(move_vectors) == len(train_pairs) and len(move_vectors) > 0:
        # Consistent movement vector across all pairs
        drs = [v[0] for v in move_vectors]
        dcs = [v[1] for v in move_vectors]
        if np.std(drs) < 0.1 and np.std(dcs) < 0.1:
            hypotheses['object_move'] = {
                'dr': float(np.mean(drs)),
                'dc': float(np.mean(dcs)),
                'confidence': 1.0
            }

    # Hypothesis: count encoding
    count_result = detect_count_rule(train_pairs)
    if count_result:
        tc, bc, conf = count_result
        hypotheses['count_encode'] = {
            'target_color': tc,
            'bar_color': bc,
            'confidence': conf
        }

    return hypotheses


def apply_causal_transform(inp, hypotheses):
    """Apply the best causal hypothesis to produce output."""
    if 'object_move' in hypotheses:
        h_data = hypotheses['object_move']
        dr = h_data['dr']
        dc = h_data['dc']
        objs = find_objects(inp)
        if len(objs) >= 1:
            h, w = inp.shape
            out = np.zeros_like(inp)
            for obj in objs:
                for (r, c) in obj['cells']:
                    nr = int(round(r + dr))
                    nc = int(round(c + dc))
                    if 0 <= nr < h and 0 <= nc < w:
                        out[nr, nc] = obj['color']
            return out

    if 'count_encode' in hypotheses:
        h_data = hypotheses['count_encode']
        return apply_count_rule(inp, h_data['target_color'], h_data['bar_color'])

    return None


# ══════════════════════════════════════════════════════════════════════════════
# PRAGMA v0.1: The Unified Three-Pillar System
# ══════════════════════════════════════════════════════════════════════════════

class PRAGMAv01:
    """
    PRAGMA v0.1 — Three-Pillar Core.

    Routing logic (Bayesian posterior entropy as gating signal):
      Tier 1: entropy < 0.5  → Symbolic certain. Use it directly.
      Tier 2: entropy 0.5–2.0 → Ambiguous. Run causal module (object tracker
              + counting). If causal finds a hypothesis, use it.
      Tier 3: entropy > 2.0  → No rule found. Neural generalization.

    This is the first system where all three pillars talk to each other
    through a shared uncertainty signal, not just output voting.
    """

    def __init__(self,
                 tier1_threshold=0.5,
                 tier2_threshold=2.0):
        self.t1 = tier1_threshold
        self.t2 = tier2_threshold
        self._symbolic = SymbolicBaseline(confidence_threshold=0.85)
        self._bayesian  = BayesianBaseline(temperature=2.0)
        self._neural    = NeuralBaseline(k=3)

    def solve(self, task):
        # ── Get Bayesian uncertainty estimate ─────────────────────────────
        bay_pred, unc = self._bayesian.solve_with_uncertainty(task)
        entropy = unc['entropy']
        bay_conf = unc['map_confidence']

        # ── TIER 1: Symbolic certain ──────────────────────────────────────
        if entropy < self.t1:
            sym_pred, expl = self._symbolic.solve_with_explanation(task)
            if sym_pred is not None and expl['confidence'] >= 0.85:
                return sym_pred, 'tier1_symbolic', entropy

        # ── TIER 2: Causal module ─────────────────────────────────────────
        if entropy < self.t2:
            # ALWAYS run causal module first — it detects object_move and
            # count_encode which symbolic mislabels with high confidence
            causal_hyps = detect_causal_transform(task.train_pairs)
            if causal_hyps:
                causal_pred = apply_causal_transform(task.test_input, causal_hyps)
                if causal_pred is not None:
                    return causal_pred, 'tier2_causal', entropy

            # No causal structure found → symbolic
            sym_pred, expl = self._symbolic.solve_with_explanation(task)
            if sym_pred is not None and expl['confidence'] >= 0.85:
                return sym_pred, 'tier2_symbolic', entropy

            # Bayesian MAP as last resort in tier 2
            if bay_pred is not None:
                return bay_pred, 'tier2_bayesian', entropy

        # ── TIER 3: Neural generalization ─────────────────────────────────
        neu_pred = self._neural.solve(task)
        if neu_pred is not None:
            return neu_pred, 'tier3_neural', entropy

        # ── Fallback ──────────────────────────────────────────────────────
        return task.test_input.copy(), 'fallback_identity', entropy

    def solve_batch(self, tasks):
        preds, tiers = [], {'tier1_symbolic':0,'tier2_symbolic':0,
                            'tier2_causal':0,'tier2_bayesian':0,
                            'tier3_neural':0,'fallback_identity':0}
        for task in tasks:
            pred, tier, entropy = self.solve(task)
            preds.append(pred)
            tiers[tier] = tiers.get(tier, 0) + 1
        return preds, tiers


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_experiment():
    print("=" * 70)
    print("PRAGMA PHASE 3 — PRAGMA v0.1: THREE-PILLAR CORE")
    print("=" * 70)
    print()
    print("New modules: Object Tracker | Counting | Causal Delta Detector")
    print("Target: 0.95+ exact match")
    print()

    tasks = generate_benchmark(n_tasks_per_type=10, seed=42)
    task_types = sorted(set(t.name for t in tasks))
    print(f"Loaded {len(tasks)} tasks\n")

    # ── Phase 1+2 References ─────────────────────────────────────────────
    print("Running reference baselines...")
    sym_preds = SymbolicBaseline().solve_batch(tasks)
    neu_preds = NeuralBaseline().solve_batch(tasks)
    sym_res   = evaluate_baseline(tasks, sym_preds)
    neu_res   = evaluate_baseline(tasks, neu_preds)

    # Best Phase 2 system: NeuSym
    from pragma.experiments.phase2_exp1_neusym import NeuSymBaseline
    neusym_preds, _ = NeuSymBaseline(fusion_mode='confidence_gated').solve_batch(tasks)
    neusym_res = evaluate_baseline(tasks, neusym_preds)
    print(f"  Symbolic:  EM={sym_res['exact_match']:.3f}")
    print(f"  NeuSym:    EM={neusym_res['exact_match']:.3f}  (best Phase 2)")
    print()

    # ── PRAGMA v0.1 ───────────────────────────────────────────────────────
    print("Running PRAGMA v0.1...")
    pragma = PRAGMAv01(tier1_threshold=0.5, tier2_threshold=2.0)
    t0 = time.time()
    pragma_preds, tiers = pragma.solve_batch(tasks)
    elapsed = time.time() - t0
    pragma_res = evaluate_baseline(tasks, pragma_preds)

    print(f"  Done in {elapsed:.3f}s")
    print(f"  Exact Match:    {pragma_res['exact_match']:.3f}")
    print(f"  Pixel Accuracy: {pragma_res['pixel_accuracy']:.3f}")
    print(f"  Tier routing:   {tiers}")
    print()

    # ── Per Task Type ─────────────────────────────────────────────────────
    print(f"  {'Task Type':<22} {'Symbolic':>9} {'NeuSym':>8} {'PRAGMA':>8} {'Delta':>7}  Status")
    print("  " + "─" * 70)
    solved_new = []
    for tt in task_types:
        s  = sym_res['by_task_type'].get(tt, 0.0)
        ns = neusym_res['by_task_type'].get(tt, 0.0)
        p  = pragma_res['by_task_type'].get(tt, 0.0)
        delta = p - max(s, ns)
        if delta > 0.05 and max(s,ns) < 0.5:
            status = "★ NEWLY SOLVED"
            solved_new.append(tt)
        elif delta > 0.05:
            status = "↑ improved"
        elif delta < -0.05:
            status = "↓ regressed"
        else:
            status = ""
        print(f"  {tt:<22} {s:>9.2f} {ns:>8.2f} {p:>8.2f} {delta:>+6.2f}   {status}")

    print()

    # ── Summary ───────────────────────────────────────────────────────────
    print("=" * 70)
    print("PHASE 3 RESULTS")
    print("=" * 70)
    methods = {
        'Neural (P1)':     neu_res['exact_match'],
        'Symbolic (P1)':   sym_res['exact_match'],
        'NeuSym (P2)':     neusym_res['exact_match'],
        'PRAGMA v0.1':     pragma_res['exact_match'],
    }
    for method, em in sorted(methods.items(), key=lambda x: x[1]):
        bar = '█' * int(em * 40)
        print(f"  {method:<18} {em:.3f}  {bar}")
    print()

    best_prev = neusym_res['exact_match']
    improvement = pragma_res['exact_match'] - best_prev
    target_hit = pragma_res['exact_match'] >= 0.95

    print(f"  Improvement over NeuSym: {improvement:+.3f}")
    print(f"  Target (0.95):           {'✓ HIT' if target_hit else '✗ MISSED — see analysis below'}")
    print()

    if solved_new:
        print(f"  Newly solved tasks: {solved_new}")
    else:
        remaining = [tt for tt in task_types
                     if pragma_res['by_task_type'].get(tt, 0) < 0.5]
        print(f"  Remaining hard tasks: {remaining}")
        print(f"  These need Phase 3.5 targeted fixes (see analysis)")

    print()
    print("  Tier routing breakdown:")
    total = sum(tiers.values())
    for tier, count in sorted(tiers.items(), key=lambda x: -x[1]):
        if count > 0:
            pct = count/total*100
            print(f"    {tier:<25} {count:3d} tasks ({pct:.0f}%)")

    print()
    print("SEND THIS ENTIRE OUTPUT BACK.")
    print("=" * 70)

    # ── Save & Plot ───────────────────────────────────────────────────────
    os.makedirs(_RESULTS, exist_ok=True)
    results_data = {
        'symbolic': sym_res, 'neural': neu_res,
        'neusym': neusym_res, 'pragma_v01': pragma_res,
        'tiers': tiers, 'elapsed': elapsed,
        'improvement': improvement, 'target_hit': target_hit
    }
    with open(os.path.join(_RESULTS, 'phase3_pragma_v01.json'), 'w') as f:
        json.dump(results_data, f, indent=2, default=str)

    _plot_results(task_types, sym_res, neusym_res, pragma_res, methods, tiers)


def _plot_results(task_types, sym_res, neusym_res, pragma_res, methods, tiers):
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle("PRAGMA v0.1 — Phase 3 Results", fontsize=15, fontweight='bold')

    # Top left: overall progression
    ax1 = fig.add_subplot(2, 3, 1)
    names = list(methods.keys())
    vals  = [methods[n] for n in names]
    colors = ['#2196F3','#4CAF50','#E91E63','#FF6F00']
    bars = ax1.bar(names, vals, color=colors, alpha=0.9, edgecolor='white')
    ax1.set_ylim(0, 1.1)
    ax1.axhline(0.95, color='gold', linestyle='--', linewidth=2, label='Target 0.95')
    ax1.legend(fontsize=9)
    for bar, val in zip(bars, vals):
        ax1.text(bar.get_x()+bar.get_width()/2, val+0.01,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax1.set_title("Overall Accuracy Progression", fontweight='bold')
    ax1.set_ylabel("Exact Match")
    ax1.tick_params(axis='x', rotation=20)
    ax1.spines[['top','right']].set_visible(False)

    # Top middle: per-task-type comparison
    ax2 = fig.add_subplot(2, 3, (2, 3))
    x = np.arange(len(task_types)); w = 0.28
    s_vals  = [sym_res['by_task_type'].get(tt,0) for tt in task_types]
    ns_vals = [neusym_res['by_task_type'].get(tt,0) for tt in task_types]
    p_vals  = [pragma_res['by_task_type'].get(tt,0) for tt in task_types]
    ax2.bar(x-w,   s_vals,  w, label='Symbolic (P1)',  color='#4CAF50', alpha=0.7)
    ax2.bar(x,     ns_vals, w, label='NeuSym (P2)',    color='#E91E63', alpha=0.7)
    ax2.bar(x+w,   p_vals,  w, label='PRAGMA v0.1',   color='#FF6F00', alpha=0.9, edgecolor='white')
    ax2.set_xticks(x)
    ax2.set_xticklabels([t.replace('_','\n') for t in task_types], fontsize=8)
    ax2.set_ylim(0, 1.25); ax2.set_ylabel("Exact Match"); ax2.legend(fontsize=9)
    ax2.set_title("Per-Task-Type: Symbolic → NeuSym → PRAGMA v0.1", fontweight='bold')
    ax2.spines[['top','right']].set_visible(False)

    # Bottom left: difficulty analysis
    ax3 = fig.add_subplot(2, 3, 4)
    diffs = ['diff_1','diff_2','diff_3']; dlabels = ['Easy','Medium','Hard']
    x3 = np.arange(3); w3 = 0.28
    ax3.bar(x3-w3, [sym_res['by_difficulty'].get(d,0) for d in diffs],
            w3, label='Symbolic', color='#4CAF50', alpha=0.8)
    ax3.bar(x3,    [neusym_res['by_difficulty'].get(d,0) for d in diffs],
            w3, label='NeuSym', color='#E91E63', alpha=0.8)
    ax3.bar(x3+w3, [pragma_res['by_difficulty'].get(d,0) for d in diffs],
            w3, label='PRAGMA v0.1', color='#FF6F00', alpha=0.9)
    ax3.set_xticks(x3); ax3.set_xticklabels(dlabels)
    ax3.set_ylim(0,1.15); ax3.set_ylabel("Exact Match"); ax3.legend(fontsize=9)
    ax3.set_title("Performance by Difficulty", fontweight='bold')
    ax3.spines[['top','right']].set_visible(False)

    # Bottom middle: tier routing pie
    ax4 = fig.add_subplot(2, 3, 5)
    tier_labels = [k.replace('_',' ') for k, v in tiers.items() if v > 0]
    tier_vals   = [v for v in tiers.values() if v > 0]
    tier_colors = ['#4CAF50','#2196F3','#FF6F00','#FF9800','#9C27B0','#607D8B']
    ax4.pie(tier_vals, labels=tier_labels, colors=tier_colors[:len(tier_vals)],
            autopct='%1.0f%%', startangle=90, textprops={'fontsize': 8})
    ax4.set_title("How PRAGMA Routes Tasks\n(which pillar handles what)", fontweight='bold')

    # Bottom right: delta heatmap
    ax5 = fig.add_subplot(2, 3, 6)
    deltas = np.array([[pragma_res['by_task_type'].get(tt,0) - sym_res['by_task_type'].get(tt,0)]
                        for tt in task_types]).T
    im = ax5.imshow(deltas, cmap='RdYlGn', vmin=-0.5, vmax=0.5, aspect='auto')
    ax5.set_xticks(range(len(task_types)))
    ax5.set_xticklabels([t.replace('_','\n') for t in task_types], fontsize=7)
    ax5.set_yticks([0]); ax5.set_yticklabels(['PRAGMA\nvs Symbolic'])
    ax5.set_title("Improvement over Symbolic\n(green=better, red=worse)", fontweight='bold')
    for j in range(len(task_types)):
        ax5.text(j, 0, f"{deltas[0,j]:+.1f}", ha='center', va='center',
                 fontsize=7, fontweight='bold')
    plt.colorbar(im, ax=ax5)

    plt.tight_layout()
    plt.savefig(os.path.join(_RESULTS, 'phase3_pragma_v01.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n  Figure saved to: {_RESULTS}/phase3_pragma_v01.png")


if __name__ == "__main__":
    run_experiment()
