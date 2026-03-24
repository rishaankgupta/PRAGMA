"""
PRAGMA Phase 1 Benchmark: Abstract Reasoning Corpus (ARC-style)
================================================================
Generates synthetic abstract reasoning tasks covering:
  - Color mapping / substitution
  - Geometric transformation (rotate, reflect, translate)
  - Pattern completion (symmetry)
  - Object counting → encoded output
  - Gravity / physics simulation
  - Rule chaining (multi-step transformations)

Each task: N training (input, output) pairs + 1 test input.
System must infer the latent rule and produce the correct test output.
"""

import numpy as np
from copy import deepcopy

COLORS = list(range(10))  # 0=black background, 1-9 = colors

class Task:
    """One abstract reasoning task."""
    def __init__(self, name, train_pairs, test_input, test_output, rule_description, difficulty):
        self.name = name
        self.train_pairs = train_pairs        # list of (input_grid, output_grid)
        self.test_input = test_input
        self.test_output = test_output        # ground truth (hidden from solver)
        self.rule_description = rule_description
        self.difficulty = difficulty          # 1=easy, 2=medium, 3=hard

    def __repr__(self):
        return f"Task({self.name}, diff={self.difficulty}, train={len(self.train_pairs)})"


def _rand_grid(h, w, bg=0, n_colors=3, density=0.3, rng=None):
    """Random grid with a few colored objects on a background."""
    if rng is None: rng = np.random.default_rng()
    grid = np.full((h, w), bg, dtype=np.int32)
    colors = rng.choice([c for c in COLORS if c != bg], size=n_colors, replace=False)
    for _ in range(int(h * w * density)):
        r, c = rng.integers(0, h), rng.integers(0, w)
        grid[r, c] = rng.choice(colors)
    return grid, colors


# ─── TASK FAMILIES ────────────────────────────────────────────────────────────

def make_color_swap_task(rng, n_train=3):
    """Rule: swap two specific colors everywhere in the grid."""
    h, w = rng.integers(4, 9), rng.integers(4, 9)
    ca, cb = rng.choice([c for c in COLORS[1:]], size=2, replace=False)
    pairs = []
    for _ in range(n_train + 1):
        inp, _ = _rand_grid(h, w, bg=0, n_colors=3, density=0.35, rng=rng)
        # Force ca and cb to appear
        inp[rng.integers(0, h), rng.integers(0, w)] = ca
        inp[rng.integers(0, h), rng.integers(0, w)] = cb
        out = inp.copy()
        out[inp == ca] = cb
        out[inp == cb] = ca
        pairs.append((inp, out))
    train, test = pairs[:n_train], pairs[n_train]
    return Task("color_swap", train, test[0], test[1],
                f"Swap color {ca} ↔ {cb}", difficulty=1)


def make_mirror_task(rng, n_train=3):
    """Rule: reflect the grid horizontally."""
    h, w = rng.integers(4, 9), rng.integers(4, 9)
    pairs = []
    for _ in range(n_train + 1):
        inp, _ = _rand_grid(h, w, bg=0, n_colors=3, density=0.4, rng=rng)
        out = np.fliplr(inp)
        pairs.append((inp, out))
    train, test = pairs[:n_train], pairs[n_train]
    return Task("mirror_horizontal", train, test[0], test[1],
                "Reflect grid horizontally", difficulty=1)


def make_rotate90_task(rng, n_train=3):
    """Rule: rotate 90 degrees clockwise."""
    pairs = []
    for _ in range(n_train + 1):
        h, w = rng.integers(3, 7), rng.integers(3, 7)
        inp, _ = _rand_grid(h, w, bg=0, n_colors=3, density=0.4, rng=rng)
        out = np.rot90(inp, k=-1)  # clockwise
        pairs.append((inp, out))
    train, test = pairs[:n_train], pairs[n_train]
    return Task("rotate_90cw", train, test[0], test[1],
                "Rotate 90° clockwise", difficulty=1)


def make_border_task(rng, n_train=3):
    """Rule: fill the border cells with a specific color, leave interior unchanged."""
    h, w = rng.integers(4, 9), rng.integers(4, 9)
    border_color = int(rng.integers(1, 10))
    pairs = []
    for _ in range(n_train + 1):
        inp, _ = _rand_grid(h, w, bg=0, n_colors=2, density=0.3, rng=rng)
        out = inp.copy()
        out[0, :] = border_color; out[-1, :] = border_color
        out[:, 0] = border_color; out[:, -1] = border_color
        pairs.append((inp, out))
    train, test = pairs[:n_train], pairs[n_train]
    return Task("fill_border", train, test[0], test[1],
                f"Fill border with color {border_color}", difficulty=1)


def make_diagonal_task(rng, n_train=3):
    """Rule: set main diagonal to a fixed color."""
    n = int(rng.integers(5, 9))
    diag_color = int(rng.integers(1, 10))
    pairs = []
    for _ in range(n_train + 1):
        inp, _ = _rand_grid(n, n, bg=0, n_colors=3, density=0.3, rng=rng)
        out = inp.copy()
        for i in range(n): out[i, i] = diag_color
        pairs.append((inp, out))
    train, test = pairs[:n_train], pairs[n_train]
    return Task("fill_diagonal", train, test[0], test[1],
                f"Set diagonal to color {diag_color}", difficulty=2)


def make_gravity_task(rng, n_train=3):
    """Rule: drop all non-background cells to the bottom of their column."""
    h, w = rng.integers(5, 9), rng.integers(4, 8)
    pairs = []
    for _ in range(n_train + 1):
        inp, _ = _rand_grid(h, w, bg=0, n_colors=3, density=0.25, rng=rng)
        out = np.zeros_like(inp)
        for col in range(w):
            items = [inp[r, col] for r in range(h) if inp[r, col] != 0]
            for i, val in enumerate(reversed(items)):
                out[h - 1 - i, col] = val
        pairs.append((inp, out))
    train, test = pairs[:n_train], pairs[n_train]
    return Task("gravity_down", train, test[0], test[1],
                "Drop all colored cells to column bottoms", difficulty=2)


def make_symmetry_completion_task(rng, n_train=3):
    """Rule: make grid vertically symmetric by mirroring top half to bottom."""
    w = int(rng.integers(5, 9))
    h = 2 * int(rng.integers(2, 4))  # even height
    half = h // 2
    pairs = []
    for _ in range(n_train + 1):
        top, _ = _rand_grid(half, w, bg=0, n_colors=3, density=0.4, rng=rng)
        inp = np.zeros((h, w), dtype=np.int32)
        inp[:half] = top
        out = inp.copy()
        out[half:] = np.flipud(top)
        pairs.append((inp, out))
    train, test = pairs[:n_train], pairs[n_train]
    return Task("vertical_symmetry", train, test[0], test[1],
                "Mirror top half to bottom half", difficulty=2)


def make_count_encode_task(rng, n_train=4):
    """Rule: count occurrences of color 1, output is a 1×N bar of that length."""
    pairs = []
    h, w = int(rng.integers(4, 8)), int(rng.integers(4, 8))
    for _ in range(n_train + 1):
        inp, _ = _rand_grid(h, w, bg=0, n_colors=2, density=0.3, rng=rng)
        count = int(np.sum(inp == 1))
        count = max(1, min(count, w))  # clamp to grid width
        out = np.zeros((1, w), dtype=np.int32)
        out[0, :count] = 1
        pairs.append((inp, out))
    train, test = pairs[:n_train], pairs[n_train]
    return Task("count_encode", train, test[0], test[1],
                "Count color-1 cells; output is a bar of that length", difficulty=2)


def make_color_chain_task(rng, n_train=4):
    """Rule: apply two sequential color substitutions (chain rule)."""
    h, w = int(rng.integers(4, 8)), int(rng.integers(4, 8))
    # Rule: A → B → C (apply A→B first, then B→C, net effect A→C, B→C)
    ca, cb, cc = rng.choice([c for c in COLORS[1:]], size=3, replace=False)
    pairs = []
    for _ in range(n_train + 1):
        inp, _ = _rand_grid(h, w, bg=0, n_colors=3, density=0.35, rng=rng)
        inp[rng.integers(0,h), rng.integers(0,w)] = ca
        inp[rng.integers(0,h), rng.integers(0,w)] = cb
        out = inp.copy()
        out[inp == ca] = cc  # A→C
        out[inp == cb] = cc  # B→C
        pairs.append((inp, out))
    train, test = pairs[:n_train], pairs[n_train]
    return Task("color_chain", train, test[0], test[1],
                f"Map colors {ca} and {cb} both to {cc}", difficulty=3)


def make_object_move_task(rng, n_train=3):
    """Rule: move the single non-background object one step in a fixed direction."""
    h, w = int(rng.integers(5, 9)), int(rng.integers(5, 9))
    dr, dc = rng.choice([-1, 0, 1]), rng.choice([-1, 0, 1])
    while dr == 0 and dc == 0:
        dr, dc = rng.choice([-1, 0, 1]), rng.choice([-1, 0, 1])
    color = int(rng.integers(1, 10))
    pairs = []
    for _ in range(n_train + 1):
        inp = np.zeros((h, w), dtype=np.int32)
        r0, c0 = int(rng.integers(1, h-1)), int(rng.integers(1, w-1))
        inp[r0, c0] = color
        out = np.zeros((h, w), dtype=np.int32)
        nr, nc = np.clip(r0 + dr, 0, h-1), np.clip(c0 + dc, 0, w-1)
        out[nr, nc] = color
        pairs.append((inp, out))
    train, test = pairs[:n_train], pairs[n_train]
    direction = {(-1,-1):"↖",(-1,0):"↑",(-1,1):"↗",(0,-1):"←",(0,1):"→",(1,-1):"↙",(1,0):"↓",(1,1):"↘"}
    return Task("object_move", train, test[0], test[1],
                f"Move object {direction.get((dr,dc),'?')} by 1 step", difficulty=3)


# ─── BENCHMARK SUITE ──────────────────────────────────────────────────────────

TASK_FACTORIES = [
    make_color_swap_task,
    make_mirror_task,
    make_rotate90_task,
    make_border_task,
    make_diagonal_task,
    make_gravity_task,
    make_symmetry_completion_task,
    make_count_encode_task,
    make_color_chain_task,
    make_object_move_task,
]

def generate_benchmark(n_tasks_per_type=10, seed=42):
    """Generate the full PRAGMA Phase 1 benchmark.
    Returns: list of Task objects.
    """
    rng = np.random.default_rng(seed)
    tasks = []
    for factory in TASK_FACTORIES:
        for i in range(n_tasks_per_type):
            try:
                task = factory(rng)
                tasks.append(task)
            except Exception as e:
                print(f"Warning: {factory.__name__} instance {i} failed: {e}")
    return tasks
