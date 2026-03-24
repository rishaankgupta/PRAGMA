"""
PRAGMA Baseline 1: Neural Approach
====================================
Architecture: Feature extraction → k-NN pattern matching
  - Extract rich feature vectors from input grids (color histograms,
    spatial frequency, local patch statistics, edge density)
  - For each test input, find the most similar training input
  - Apply the corresponding training transformation as a "learned" rule
  - Represent as the closest neural analogue without PyTorch:
    attention-weighted feature matching + linear transformation search

This mimics what a CNN/transformer does: learn a mapping from
feature space → transformation, then generalize.
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def extract_features(grid):
    """Extract a fixed-size feature vector from a grid (independent of grid size)."""
    feats = []
    h, w = grid.shape

    # 1. Color histogram (10 colors) — 10 features
    for c in range(10):
        feats.append(float(np.sum(grid == c)) / (h * w))

    # 2. Row and column entropy — summarized into fixed 4 stats each
    row_entropies = []
    for row in grid:
        hist = np.bincount(row, minlength=10) / w
        row_entropies.append(float(-np.sum(hist * np.log(hist + 1e-9))))
    feats += [np.mean(row_entropies), np.std(row_entropies),
              np.min(row_entropies), np.max(row_entropies)]

    col_entropies = []
    for col in grid.T:
        hist = np.bincount(col, minlength=10) / h
        col_entropies.append(float(-np.sum(hist * np.log(hist + 1e-9))))
    feats += [np.mean(col_entropies), np.std(col_entropies),
              np.min(col_entropies), np.max(col_entropies)]

    # 3. Spatial moments (center of mass per color)
    rows_idx, cols_idx = np.mgrid[0:h, 0:w]
    for c in range(1, 10):
        mask = (grid == c).astype(float)
        total = mask.sum()
        if total > 0:
            feats.append(float((mask * rows_idx).sum() / total) / h)
            feats.append(float((mask * cols_idx).sum() / total) / w)
        else:
            feats.append(0.5)
            feats.append(0.5)

    # 4. Edge density (difference between adjacent cells)
    h_edges = float(np.mean(grid[1:, :] != grid[:-1, :]))
    v_edges = float(np.mean(grid[:, 1:] != grid[:, :-1]))
    feats += [h_edges, v_edges]

    # 5. Symmetry scores
    lr_sym = float(np.mean(grid == np.fliplr(grid)))
    ud_sym = float(np.mean(grid == np.flipud(grid)))
    feats += [lr_sym, ud_sym]

    # 6. Grid dimensions (normalized)
    feats += [h / 30.0, w / 30.0]

    return np.array(feats, dtype=np.float32)


def grid_to_flat(grid, max_h=30, max_w=30):
    """Flatten grid to fixed-size vector with zero-padding."""
    out = np.zeros(max_h * max_w, dtype=np.int32)
    h, w = grid.shape
    for r in range(min(h, max_h)):
        for c in range(min(w, max_w)):
            out[r * max_w + c] = grid[r, c]
    return out


def extract_transformation_delta(inp, out):
    """Characterize the transformation as a feature vector."""
    delta = []
    h_in, w_in = inp.shape
    h_out, w_out = out.shape

    # Shape change
    delta.append(float(h_out - h_in))
    delta.append(float(w_out - w_in))

    # Color distribution change
    for c in range(10):
        before = float(np.sum(inp == c)) / inp.size
        after = float(np.sum(out == c)) / out.size
        delta.append(after - before)

    # Spatial relationship
    if inp.shape == out.shape:
        delta.append(float(np.mean(inp == out)))          # preservation rate
        delta.append(float(np.mean(inp == np.fliplr(out)))) # lr flip similarity
        delta.append(float(np.mean(inp == np.flipud(out)))) # ud flip similarity
        try:
            delta.append(float(np.mean(inp == np.rot90(out, k=1)))) # rot similarity
        except: delta.append(0.0)
    else:
        delta += [0.0, 0.0, 0.0, 0.0]

    return np.array(delta, dtype=np.float32)


def apply_learned_transformation(test_inp, ref_inp, ref_out):
    """
    Given a reference (inp→out) pair, try to apply the same transformation
    to test_inp. We try all atomic transformations and pick the most likely.
    This is the 'generalization' step.
    """
    h_in, w_in = ref_inp.shape
    h_out, w_out = ref_out.shape
    h_t, w_t = test_inp.shape

    candidates = []

    # Try: identity
    candidates.append(('identity', test_inp.copy()))

    # Try: color swaps (detect from ref pair)
    color_map = {}
    if ref_inp.shape == ref_out.shape:
        for c in range(10):
            mask = ref_inp == c
            if mask.any():
                out_colors = ref_out[mask]
                vals, cnts = np.unique(out_colors, return_counts=True)
                mapped = vals[np.argmax(cnts)]
                if mapped != c:
                    color_map[c] = int(mapped)

    if color_map:
        mapped = test_inp.copy()
        for src, dst in color_map.items():
            mapped[test_inp == src] = dst
        candidates.append(('color_map', mapped))

    # Try: geometric transforms
    candidates.append(('fliplr', np.fliplr(test_inp)))
    candidates.append(('flipud', np.flipud(test_inp)))
    candidates.append(('rot90cw', np.rot90(test_inp, k=-1)))
    candidates.append(('rot90ccw', np.rot90(test_inp, k=1)))
    candidates.append(('rot180', np.rot90(test_inp, k=2)))

    # Try: border fill (detect border color from ref)
    if h_t >= 2 and w_t >= 2:
        if ref_inp.shape == ref_out.shape:
            border_colors = np.concatenate([
                ref_out[0,:], ref_out[-1,:], ref_out[:,0], ref_out[:,-1]
            ])
            vals, cnts = np.unique(border_colors, return_counts=True)
            bc = vals[np.argmax(cnts)]
            bordered = test_inp.copy()
            bordered[0,:] = bc; bordered[-1,:] = bc
            bordered[:,0] = bc; bordered[:,-1] = bc
            candidates.append(('border_fill', bordered))

    # Try: gravity (drop cells down)
    gravity = test_inp.copy()
    gh, gw = gravity.shape
    for col in range(gw):
        items = [gravity[r, col] for r in range(gh) if gravity[r, col] != 0]
        col_out = np.zeros(gh, dtype=np.int32)
        for i, val in enumerate(reversed(items)):
            col_out[gh - 1 - i] = val
        gravity[:, col] = col_out
    candidates.append(('gravity', gravity))

    # Try: diagonal fill (detect from ref)
    if ref_inp.shape == ref_out.shape and ref_inp.shape[0] == ref_inp.shape[1]:
        n = ref_inp.shape[0]
        diag_colors = [ref_out[i,i] for i in range(n)]
        if len(set(diag_colors)) == 1:
            dc = diag_colors[0]
            diag_out = test_inp.copy()
            for i in range(min(diag_out.shape)):
                diag_out[i,i] = dc
            candidates.append(('diagonal', diag_out))

    # Score candidates against reference pair (pick best match)
    if ref_inp.shape == ref_out.shape:
        ref_feat_out = extract_features(ref_out)
        best_score = -1
        best_cand = candidates[0][1]
        for name, cand in candidates:
            if cand.shape == test_inp.shape:  # same shape constraint
                feat = extract_features(cand)
                # Compare transformed candidate features to ref output features
                score = float(np.dot(feat, ref_feat_out) / (np.linalg.norm(feat) * np.linalg.norm(ref_feat_out) + 1e-9))
                if score > best_score:
                    best_score = score
                    best_cand = cand
        return best_cand
    else:
        # Output shape changes: try count_encode style (1×W bar)
        count_1 = int(np.sum(test_inp == 1))
        count_1 = max(1, min(count_1, test_inp.shape[1]))
        bar = np.zeros((1, test_inp.shape[1]), dtype=np.int32)
        bar[0, :count_1] = 1
        return bar


class NeuralBaseline:
    """
    Neural baseline: feature extraction + k-NN matching + transformation transfer.
    Represents the 'pattern recognition without reasoning' approach.
    """
    def __init__(self, k=3):
        self.k = k
        self.tasks_seen = []

    def solve(self, task):
        """
        Solve one task using only the task's own training pairs
        (few-shot learning — same constraint as ARC benchmark).
        """
        train = task.train_pairs
        test_inp = task.test_input

        if not train:
            return None

        # Extract features for all training inputs
        train_feats = np.array([extract_features(inp) for inp, _ in train])
        test_feat = extract_features(test_inp)

        # Find k nearest training inputs by cosine similarity
        sims = []
        for tf in train_feats:
            sim = np.dot(test_feat, tf) / (np.linalg.norm(test_feat) * np.linalg.norm(tf) + 1e-9)
            sims.append(float(sim))

        # Use top-k most similar training pairs
        k = min(self.k, len(train))
        top_k_idx = np.argsort(sims)[::-1][:k]

        # Apply transformation from best matching training pair
        best_idx = top_k_idx[0]
        ref_inp, ref_out = train[best_idx]
        prediction = apply_learned_transformation(test_inp, ref_inp, ref_out)

        return prediction

    def solve_batch(self, tasks):
        """Solve all tasks. Returns list of predictions."""
        return [self.solve(t) for t in tasks]
