"""
PRAGMA Baseline 3: Bayesian AI Approach
=========================================
Architecture: Bayesian inference over rule hypotheses
  - Prior: uniform distribution over all candidate rules
  - Likelihood: P(observations | rule) = product of match scores per pair
  - Posterior: proportional to prior × likelihood (Bayes' theorem)
  - Prediction: MAP estimate (most likely rule) + uncertainty quantification
  - Uncertainty output: probability distribution over output cells

This baseline produces calibrated uncertainty estimates — the key
differentiator from symbolic (deterministic) and neural (point estimate) approaches.
"""

import numpy as np
from pragma.baselines.symbolic import RULES


class BayesianBaseline:
    """
    Bayesian inference over a discrete hypothesis space of transformation rules.
    Uses Bayes' theorem: P(rule | data) ∝ P(data | rule) × P(rule)
    Key output: posterior uncertainty over predictions, measured via Brier score.
    """
    def __init__(self, prior_type="uniform", temperature=2.0):
        self.prior_type = prior_type
        self.temperature = temperature  # sharpness of likelihood

    def _compute_posterior(self, train_pairs):
        """Compute posterior P(rule | training pairs) for all rules."""
        n_rules = len(RULES)

        # ── Prior ──────────────────────────────────────────────────────────
        if self.prior_type == "uniform":
            log_prior = np.zeros(n_rules)
        elif self.prior_type == "geometric_bias":
            # Prefer simpler rules (color swap, geometric) over complex ones
            complexity = [1,1,1,1,1,1,2,2,2,2,3,0]  # 0=identity=simplest
            log_prior = -np.array(complexity[:n_rules], dtype=float) * 0.5
        else:
            log_prior = np.zeros(n_rules)

        # ── Likelihood ─────────────────────────────────────────────────────
        log_likelihood = np.zeros(n_rules)
        for i, (name, detector, applicator) in enumerate(RULES):
            try:
                confidence, params = detector(train_pairs)
                # Convert confidence to log-likelihood
                # P(data | rule) = confidence^(n_pairs * temperature)
                n = len(train_pairs)
                log_likelihood[i] = n * self.temperature * np.log(confidence + 1e-9)
            except Exception:
                log_likelihood[i] = -1e9  # impossible rule

        # ── Posterior (unnormalized log) ───────────────────────────────────
        log_posterior = log_prior + log_likelihood

        # Softmax normalization to get proper probability distribution
        log_posterior -= np.max(log_posterior)  # numerical stability
        posterior = np.exp(log_posterior)
        posterior /= posterior.sum()

        return posterior

    def _generate_prediction_distribution(self, test_inp, posterior):
        """
        Generate a probability distribution over output grids.
        For each cell (r, c), produce a distribution over colors 0-9.
        """
        h, w = test_inp.shape
        # cell_probs[r, c, color] = probability
        cell_probs = np.zeros((h, w, 10), dtype=float)
        # Start with uniform prior over colors per cell
        cell_probs[:, :, :] = 1.0 / 10

        # For each rule, compute its predicted output and weight by posterior
        for i, (name, detector, applicator) in enumerate(RULES):
            rule_weight = posterior[i]
            if rule_weight < 1e-6:
                continue
            try:
                # Get params from detector
                train_confidence, params = detector([])  # no training data for params
                pred = applicator(test_inp, params)
                if pred.shape == (h, w):
                    for r in range(h):
                        for c in range(w):
                            color = int(pred[r, c])
                            cell_probs[r, c, :] *= (1 - rule_weight)
                            cell_probs[r, c, color] += rule_weight
            except Exception:
                continue

        # Renormalize each cell
        sums = cell_probs.sum(axis=2, keepdims=True)
        cell_probs /= (sums + 1e-9)
        return cell_probs

    def solve(self, task):
        """
        Solve task using MAP (Maximum A Posteriori) rule.
        Returns the most probable prediction.
        """
        posterior = self._compute_posterior(task.train_pairs)
        best_rule_idx = int(np.argmax(posterior))
        name, detector, applicator = RULES[best_rule_idx]

        try:
            # Get params from full training set
            _, params = detector(task.train_pairs)
            pred = applicator(task.test_input, params)
            return pred
        except Exception:
            return task.test_input.copy()

    def solve_with_uncertainty(self, task):
        """
        Full Bayesian solve: returns (prediction, uncertainty_info).
        uncertainty_info contains:
          - posterior: probability over each rule
          - entropy: H[P(rule|data)] — how uncertain we are about which rule
          - cell_probs: per-cell color probability distribution
          - map_rule: name of most probable rule
          - map_confidence: posterior probability of best rule
        """
        posterior = self._compute_posterior(task.train_pairs)
        best_rule_idx = int(np.argmax(posterior))
        name, detector, applicator = RULES[best_rule_idx]

        # Compute prediction from MAP rule
        pred = None
        try:
            _, params = detector(task.train_pairs)
            pred = applicator(task.test_input, params)
        except Exception:
            pred = task.test_input.copy()

        # Posterior entropy (uncertainty about which rule applies)
        entropy = float(-np.sum(posterior * np.log(posterior + 1e-9)))

        # Per-cell probability distribution (for Brier score)
        h, w = task.test_input.shape
        cell_probs = np.zeros((h * w, 10), dtype=float)
        for i, (rname, rdetect, rapply) in enumerate(RULES):
            rw = posterior[i]
            if rw < 1e-6: continue
            try:
                _, rparams = rdetect(task.train_pairs)
                rpred = rapply(task.test_input, rparams)
                if rpred.shape == (h, w):
                    flat = rpred.flatten()
                    for j, color in enumerate(flat):
                        cell_probs[j, int(color)] += rw
            except Exception:
                pass

        # Normalize
        sums = cell_probs.sum(axis=1, keepdims=True)
        cell_probs /= (sums + 1e-9)

        return pred, {
            "posterior": posterior,
            "rule_names": [r[0] for r in RULES],
            "entropy": entropy,
            "map_rule": name,
            "map_confidence": float(posterior[best_rule_idx]),
            "cell_probs": cell_probs,  # shape (h*w, 10)
        }

    def solve_batch(self, tasks):
        return [self.solve(t) for t in tasks]

    def solve_batch_with_uncertainty(self, tasks):
        results = []
        for t in tasks:
            pred, uncertainty = self.solve_with_uncertainty(t)
            results.append((pred, uncertainty))
        return results
