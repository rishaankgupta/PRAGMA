"""
PRAGMA Baseline 2: Symbolic AI Approach
=========================================
Architecture: Rule induction from examples → rule application
  - Examine each (input, output) training pair
  - Hypothesize candidate transformation rules (symbolic predicates)
  - Verify candidate rules across ALL training pairs
  - Apply verified rule to test input

This is classical symbolic AI: explicit hypothesis generation,
verification against evidence, deterministic rule application.
Rules are human-interpretable logical predicates.
"""

import numpy as np
from itertools import product


# ─── RULE DEFINITIONS ─────────────────────────────────────────────────────────
# Each rule: (name, detector, applicator)
# detector(train_pairs) → confidence ∈ [0,1], params
# applicator(inp, params) → output grid

def _detect_color_swap(train_pairs):
    """Detect if all pairs apply a consistent color permutation."""
    if not train_pairs: return 0.0, {}
    # Try all pairs of colors (ca → cb and cb → ca)
    first_inp, first_out = train_pairs[0]
    if first_inp.shape != first_out.shape: return 0.0, {}

    # Build color mapping from first pair
    color_map = {}
    for c in range(10):
        mask = first_inp == c
        if not mask.any(): continue
        out_vals = first_out[mask]
        vals, cnts = np.unique(out_vals, return_counts=True)
        if len(vals) == 1:
            color_map[c] = int(vals[0])
        else:
            return 0.0, {}  # ambiguous mapping

    if not color_map: return 0.0, {}

    # Verify on remaining pairs
    verified = 0
    for inp, out in train_pairs:
        if inp.shape != out.shape: return 0.0, {}
        consistent = True
        for c, mapped in color_map.items():
            mask = inp == c
            if mask.any() and not np.all(out[mask] == mapped):
                consistent = False; break
        if consistent: verified += 1

    confidence = verified / len(train_pairs)
    return confidence, {"color_map": color_map}


def _apply_color_swap(inp, params):
    out = inp.copy()
    for src, dst in params["color_map"].items():
        out[inp == src] = dst
    return out


def _detect_fliplr(train_pairs):
    scores = []
    for inp, out in train_pairs:
        if inp.shape != out.shape: return 0.0, {}
        scores.append(float(np.mean(np.fliplr(inp) == out)))
    return float(np.mean(scores)), {}


def _apply_fliplr(inp, params):
    return np.fliplr(inp)


def _detect_flipud(train_pairs):
    scores = []
    for inp, out in train_pairs:
        if inp.shape != out.shape: return 0.0, {}
        scores.append(float(np.mean(np.flipud(inp) == out)))
    return float(np.mean(scores)), {}


def _apply_flipud(inp, params):
    return np.flipud(inp)


def _detect_rot90cw(train_pairs):
    scores = []
    for inp, out in train_pairs:
        try:
            rotated = np.rot90(inp, k=-1)
            if rotated.shape != out.shape: return 0.0, {}
            scores.append(float(np.mean(rotated == out)))
        except: return 0.0, {}
    return float(np.mean(scores)), {}


def _apply_rot90cw(inp, params):
    return np.rot90(inp, k=-1)


def _detect_rot90ccw(train_pairs):
    scores = []
    for inp, out in train_pairs:
        try:
            rotated = np.rot90(inp, k=1)
            if rotated.shape != out.shape: return 0.0, {}
            scores.append(float(np.mean(rotated == out)))
        except: return 0.0, {}
    return float(np.mean(scores)), {}


def _apply_rot90ccw(inp, params):
    return np.rot90(inp, k=1)


def _detect_rot180(train_pairs):
    scores = []
    for inp, out in train_pairs:
        if inp.shape != out.shape: return 0.0, {}
        scores.append(float(np.mean(np.rot90(inp, k=2) == out)))
    return float(np.mean(scores)), {}


def _apply_rot180(inp, params):
    return np.rot90(inp, k=2)


def _detect_border_fill(train_pairs):
    """Detect border fill: interior same, border changed to fixed color."""
    border_colors = []
    for inp, out in train_pairs:
        if inp.shape != out.shape: return 0.0, {}
        h, w = out.shape
        if h < 3 or w < 3: return 0.0, {}
        border = np.concatenate([out[0,:], out[-1,:], out[:,0], out[:,-1]])
        vals, cnts = np.unique(border, return_counts=True)
        bc = vals[np.argmax(cnts)]
        interior_preserved = np.mean(inp[1:-1, 1:-1] == out[1:-1, 1:-1])
        if interior_preserved < 0.9: return 0.0, {}
        border_cells_filled = np.mean(border == bc)
        if border_cells_filled < 0.9: return 0.0, {}
        border_colors.append(int(bc))
    if len(set(border_colors)) == 1:
        return 1.0, {"border_color": border_colors[0]}
    return 0.0, {}


def _apply_border_fill(inp, params):
    out = inp.copy()
    bc = params["border_color"]
    out[0,:] = bc; out[-1,:] = bc
    out[:,0] = bc; out[:,-1] = bc
    return out


def _detect_diagonal(train_pairs):
    diag_colors = []
    for inp, out in train_pairs:
        if inp.shape != out.shape: return 0.0, {}
        n = min(out.shape)
        diag = [out[i,i] for i in range(n)]
        if len(set(diag)) != 1: return 0.0, {}
        dc = diag[0]
        # Check non-diagonal is preserved
        preserved = True
        for r in range(out.shape[0]):
            for c in range(out.shape[1]):
                if r != c and inp[r,c] != out[r,c]:
                    preserved = False; break
            if not preserved: break
        if not preserved: return 0.0, {}
        diag_colors.append(int(dc))
    if len(set(diag_colors)) == 1:
        return 1.0, {"diag_color": diag_colors[0]}
    return 0.0, {}


def _apply_diagonal(inp, params):
    out = inp.copy()
    dc = params["diag_color"]
    for i in range(min(out.shape)):
        out[i,i] = dc
    return out


def _detect_gravity(train_pairs):
    scores = []
    for inp, out in train_pairs:
        if inp.shape != out.shape: return 0.0, {}
        h, w = inp.shape
        expected = np.zeros_like(inp)
        for col in range(w):
            items = [inp[r, col] for r in range(h) if inp[r, col] != 0]
            for i, val in enumerate(reversed(items)):
                expected[h - 1 - i, col] = val
        scores.append(float(np.mean(expected == out)))
    return float(np.mean(scores)), {}


def _apply_gravity(inp, params):
    h, w = inp.shape
    out = np.zeros_like(inp)
    for col in range(w):
        items = [inp[r, col] for r in range(h) if inp[r, col] != 0]
        for i, val in enumerate(reversed(items)):
            out[h - 1 - i, col] = val
    return out


def _detect_vertical_symmetry(train_pairs):
    scores = []
    for inp, out in train_pairs:
        if inp.shape != out.shape: return 0.0, {}
        h, w = inp.shape
        half = h // 2
        top_preserved = float(np.mean(inp[:half] == out[:half]))
        bottom_mirrored = float(np.mean(np.flipud(inp[:half]) == out[half:half*2]))
        scores.append((top_preserved + bottom_mirrored) / 2)
    return float(np.mean(scores)) if scores else 0.0, {}


def _apply_vertical_symmetry(inp, params):
    h, w = inp.shape
    half = h // 2
    out = inp.copy()
    out[half:half*2] = np.flipud(inp[:half])
    return out


def _detect_count_encode(train_pairs):
    """Detect: count cells of color 1, output is 1×W bar."""
    for inp, out in train_pairs:
        if out.shape[0] != 1: return 0.0, {}
        count = int(np.sum(inp == 1))
        expected = np.zeros_like(out)
        expected[0, :min(count, out.shape[1])] = 1
        if not np.array_equal(expected, out): return 0.0, {}
    return 1.0, {}


def _apply_count_encode(inp, params):
    count = int(np.sum(inp == 1))
    w = inp.shape[1]
    out = np.zeros((1, w), dtype=np.int32)
    out[0, :min(count, w)] = 1
    return out


def _detect_identity(train_pairs):
    scores = []
    for inp, out in train_pairs:
        if inp.shape != out.shape: scores.append(0.0); continue
        scores.append(float(np.mean(inp == out)))
    return float(np.mean(scores)) if scores else 0.0, {}


def _apply_identity(inp, params):
    return inp.copy()


RULES = [
    ("color_swap",         _detect_color_swap,         _apply_color_swap),
    ("fliplr",             _detect_fliplr,              _apply_fliplr),
    ("flipud",             _detect_flipud,              _apply_flipud),
    ("rot90cw",            _detect_rot90cw,             _apply_rot90cw),
    ("rot90ccw",           _detect_rot90ccw,            _apply_rot90ccw),
    ("rot180",             _detect_rot180,              _apply_rot180),
    ("border_fill",        _detect_border_fill,         _apply_border_fill),
    ("diagonal",           _detect_diagonal,            _apply_diagonal),
    ("gravity",            _detect_gravity,             _apply_gravity),
    ("vertical_symmetry",  _detect_vertical_symmetry,   _apply_vertical_symmetry),
    ("count_encode",       _detect_count_encode,        _apply_count_encode),
    ("identity",           _detect_identity,            _apply_identity),
]


class SymbolicBaseline:
    """
    Symbolic AI baseline: explicit rule induction and deduction.
    Each rule is a logical predicate that either holds or does not hold
    across all training pairs. The rule with the highest confidence is applied.
    Fully interpretable — we know exactly which rule was applied and why.
    """
    def __init__(self, confidence_threshold=0.85):
        self.threshold = confidence_threshold

    def solve(self, task):
        train = task.train_pairs
        if not train: return None

        best_rule_name = None
        best_confidence = -1.0
        best_params = {}
        best_applicator = None
        rule_scores = {}

        for name, detector, applicator in RULES:
            try:
                confidence, params = detector(train)
                rule_scores[name] = confidence
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_rule_name = name
                    best_params = params
                    best_applicator = applicator
            except Exception:
                rule_scores[name] = 0.0

        if best_confidence < self.threshold or best_applicator is None:
            # No rule found with sufficient confidence — fall back to identity
            return task.test_input.copy()

        try:
            return best_applicator(task.test_input, best_params)
        except Exception:
            return task.test_input.copy()

    def solve_with_explanation(self, task):
        """Returns (prediction, explanation_dict) for interpretability analysis."""
        train = task.train_pairs
        rule_scores = {}
        for name, detector, applicator in RULES:
            try:
                confidence, params = detector(train)
                rule_scores[name] = (confidence, params, applicator)
            except Exception:
                rule_scores[name] = (0.0, {}, None)

        best = max(rule_scores.items(), key=lambda x: x[1][0])
        rule_name = best[0]
        confidence, params, applicator = best[1]

        pred = None
        if confidence >= self.threshold and applicator:
            try:
                pred = applicator(task.test_input, params)
            except Exception:
                pred = task.test_input.copy()

        return pred, {
            "selected_rule": rule_name,
            "confidence": confidence,
            "all_scores": {k: v[0] for k, v in rule_scores.items()},
            "params": params
        }

    def solve_batch(self, tasks):
        return [self.solve(t) for t in tasks]
