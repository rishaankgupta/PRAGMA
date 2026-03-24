"""
PRAGMA Baseline 4: World Model Approach
=========================================
Architecture: Latent dynamics model + planning
  - Encoder: compress grids into low-dimensional latent vectors (PCA)
  - Dynamics model: learn transition function z_out = f(z_in) from training pairs
  - Decoder: reconstruct output grid from latent prediction
  - Planning: simulate multiple candidate transformations in latent space,
    pick the one that minimizes prediction error on training examples

This mimics DreamerV3's core loop: Encode → Predict in latent space → Decode
No backpropagation needed — we use PCA + linear regression for the core model.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


MAX_H, MAX_W = 30, 30
FLAT_DIM = MAX_H * MAX_W  # 900

def grid_to_flat(grid):
    """Pad/flatten grid to fixed-size vector."""
    flat = np.zeros(FLAT_DIM, dtype=np.float32)
    h, w = grid.shape
    h_ = min(h, MAX_H); w_ = min(w, MAX_W)
    flat[:h_ * w_] = grid[:h_, :w_].flatten().astype(np.float32)
    return flat


def flat_to_grid(flat, h, w):
    """Recover grid of shape (h, w) from flat vector."""
    h_ = min(h, MAX_H); w_ = min(w, MAX_W)
    data = np.round(flat[:h_ * w_]).astype(np.int32).reshape(h_, w_)
    data = np.clip(data, 0, 9)
    return data


def augment_features(flat):
    """Add second-order features: squared values + color indicators."""
    # Color indicator channels (one per color)
    indicators = np.array([(flat == c).astype(float).mean() for c in range(10)])
    return np.concatenate([flat, flat**2, indicators])


class WorldModelBaseline:
    """
    World model baseline: encode → dynamics → decode.
    For each task, fits a task-specific dynamics model from training pairs.
    The dynamics model is a linear map in PCA latent space.
    Planning: test against candidate transformations in latent space.
    """
    def __init__(self, latent_dim=32, alpha=0.1):
        self.latent_dim = latent_dim
        self.alpha = alpha  # Ridge regression regularization

    def _fit_task_model(self, train_pairs):
        """
        Fit a latent dynamics model for this specific task.
        Returns (pca, scaler, ridge_model, input_shapes, output_shapes)
        """
        if not train_pairs:
            return None

        # Collect input and output flat vectors
        X_flat = np.array([grid_to_flat(inp) for inp, _ in train_pairs])
        Y_flat = np.array([grid_to_flat(out) for _, out in train_pairs])

        # Augment
        X_aug = np.array([augment_features(x) for x in X_flat])
        Y_aug = np.array([augment_features(y) for y in Y_flat])

        # Fit PCA on combined input+output data to learn shared latent space
        combined = np.vstack([X_aug, Y_aug])
        n_components = min(self.latent_dim, combined.shape[0] - 1, combined.shape[1])

        scaler = StandardScaler()
        combined_scaled = scaler.fit_transform(combined)

        pca = PCA(n_components=n_components)
        pca.fit(combined_scaled)

        # Project inputs and outputs to latent space
        Z_in = pca.transform(scaler.transform(X_aug))
        Z_out = pca.transform(scaler.transform(Y_aug))

        # Fit linear dynamics: Z_out ≈ A @ Z_in
        ridge = Ridge(alpha=self.alpha)
        ridge.fit(Z_in, Z_out)

        shapes_in = [inp.shape for inp, _ in train_pairs]
        shapes_out = [out.shape for _, out in train_pairs]

        return pca, scaler, ridge, shapes_in, shapes_out

    def _predict(self, test_inp, model_bundle):
        """Apply the latent dynamics model to test input."""
        pca, scaler, ridge, shapes_in, shapes_out = model_bundle

        # Most common output shape
        from collections import Counter
        shape_counts = Counter(shapes_out)
        target_shape = shape_counts.most_common(1)[0][0]

        # Encode test input
        x_flat = grid_to_flat(test_inp)
        x_aug = augment_features(x_flat).reshape(1, -1)
        x_scaled = scaler.transform(x_aug)
        z_in = pca.transform(x_scaled)

        # Predict in latent space
        z_out = ridge.predict(z_in)

        # Decode back to grid space
        x_out_scaled = pca.inverse_transform(z_out)
        x_out = scaler.inverse_transform(x_out_scaled)
        y_aug = x_out[0]

        # Extract just the flat part (first FLAT_DIM dimensions)
        y_flat = y_aug[:FLAT_DIM]
        h, w = target_shape
        return flat_to_grid(y_flat, h, w)

    def _plan_with_candidates(self, test_inp, model_bundle, train_pairs):
        """
        Planning: generate candidate outputs from known transformations,
        evaluate each in latent space, pick best.
        This is the 'imagination' / planning phase of the world model.
        """
        pca, scaler, ridge, shapes_in, shapes_out = model_bundle

        # Candidate transformations to evaluate
        candidates = {
            'wm_prediction': self._predict(test_inp, model_bundle),
            'fliplr': np.fliplr(test_inp),
            'flipud': np.flipud(test_inp),
            'rot90cw': np.rot90(test_inp, k=-1),
            'rot90ccw': np.rot90(test_inp, k=1),
            'identity': test_inp.copy(),
        }

        # Gravity
        h, w = test_inp.shape
        grav = np.zeros_like(test_inp)
        for col in range(w):
            items = [test_inp[r, col] for r in range(h) if test_inp[r, col] != 0]
            for i, val in enumerate(reversed(items)):
                grav[h-1-i, col] = val
        candidates['gravity'] = grav

        # Border fill (most common non-zero color in training outputs)
        all_out_vals = np.concatenate([out.flatten() for _, out in train_pairs])
        nonzero_vals = all_out_vals[all_out_vals > 0]
        if len(nonzero_vals) > 0:
            bc = int(np.bincount(nonzero_vals).argmax())
            bordered = test_inp.copy()
            bordered[0,:] = bc; bordered[-1,:] = bc
            bordered[:,0] = bc; bordered[:,-1] = bc
            candidates['border_fill'] = bordered

        # Score each candidate by consistency with training dynamics in latent space
        best_name = 'wm_prediction'
        best_score = -float('inf')

        for name, cand in candidates.items():
            # Check shape compatibility
            from collections import Counter
            shape_counts = Counter(shapes_out)
            target_shape = shape_counts.most_common(1)[0][0]
            if cand.shape != target_shape: continue

            # Score: how well does this candidate match the world model's prediction?
            wm_pred = self._predict(test_inp, model_bundle)
            if wm_pred.shape == cand.shape:
                score = float(np.mean(cand == wm_pred))
            else:
                score = 0.0

            # Also score: consistency across training pairs
            train_score = 0.0
            for inp, out in train_pairs:
                try:
                    # Apply same "name" transformation to training input
                    if name == 'fliplr': tpred = np.fliplr(inp)
                    elif name == 'flipud': tpred = np.flipud(inp)
                    elif name == 'rot90cw': tpred = np.rot90(inp, k=-1)
                    elif name == 'rot90ccw': tpred = np.rot90(inp, k=1)
                    elif name == 'gravity':
                        gh, gw = inp.shape
                        tpred = np.zeros_like(inp)
                        for col in range(gw):
                            items = [inp[r, col] for r in range(gh) if inp[r, col] != 0]
                            for i, val in enumerate(reversed(items)):
                                tpred[gh-1-i, col] = val
                    elif name == 'identity': tpred = inp.copy()
                    else: continue
                    if tpred.shape == out.shape:
                        train_score += float(np.mean(tpred == out))
                except: pass
            combined_score = score * 0.3 + train_score / max(len(train_pairs), 1) * 0.7
            if combined_score > best_score:
                best_score = combined_score
                best_name = name

        return candidates.get(best_name, self._predict(test_inp, model_bundle))

    def solve(self, task):
        """Full world model solve: fit → plan → predict."""
        try:
            model = self._fit_task_model(task.train_pairs)
            if model is None: return None
            return self._plan_with_candidates(task.test_input, model, task.train_pairs)
        except Exception as e:
            try:
                model = self._fit_task_model(task.train_pairs)
                return self._predict(task.test_input, model)
            except Exception:
                return task.test_input.copy()

    def solve_batch(self, tasks):
        return [self.solve(t) for t in tasks]
