"""
PRAGMA Benchmark Evaluator
===========================
Metrics:
  - exact_match: output grid == ground truth (primary)
  - pixel_accuracy: fraction of cells correct
  - shape_match: correct output dimensions
  - uncertainty_calibration: for Bayesian baseline (Brier score)
"""

import numpy as np


def exact_match(predicted, ground_truth):
    if predicted is None: return 0.0
    if predicted.shape != ground_truth.shape: return 0.0
    return float(np.array_equal(predicted, ground_truth))


def pixel_accuracy(predicted, ground_truth):
    if predicted is None: return 0.0
    if predicted.shape != ground_truth.shape:
        # Partial credit: compare overlapping region
        min_h = min(predicted.shape[0], ground_truth.shape[0])
        min_w = min(predicted.shape[1], ground_truth.shape[1])
        pred_crop = predicted[:min_h, :min_w]
        gt_crop = ground_truth[:min_h, :min_w]
        return float(np.mean(pred_crop == gt_crop)) * 0.5  # penalize shape error
    return float(np.mean(predicted == ground_truth))


def shape_match(predicted, ground_truth):
    if predicted is None: return 0.0
    return float(predicted.shape == ground_truth.shape)


def brier_score(probs, ground_truth_flat):
    """Mean squared error between prob distribution and one-hot ground truth.
    Lower is better. probs: (n_cells, n_colors). gt: (n_cells,)
    """
    n_colors = probs.shape[1]
    one_hot = np.eye(n_colors)[ground_truth_flat.astype(int) % n_colors]
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def evaluate_baseline(tasks, predictions, uncertainties=None):
    """
    tasks: list of Task
    predictions: list of np.ndarray (or None for failure)
    uncertainties: optional list of (probs, gt_flat) tuples for Bayesian scoring
    Returns: dict of metrics
    """
    n = len(tasks)
    em_scores, pa_scores, sm_scores = [], [], []
    by_difficulty = {1: [], 2: [], 3: [], 4: []}
    by_type = {}

    for i, (task, pred) in enumerate(zip(tasks, predictions)):
        gt = task.test_output
        em = exact_match(pred, gt)
        pa = pixel_accuracy(pred, gt)
        sm = shape_match(pred, gt)
        em_scores.append(em)
        pa_scores.append(pa)
        sm_scores.append(sm)
        by_difficulty[task.difficulty].append(em)
        ttype = task.name
        by_type.setdefault(ttype, []).append(em)

    results = {
        "n_tasks": n,
        "exact_match": float(np.mean(em_scores)),
        "pixel_accuracy": float(np.mean(pa_scores)),
        "shape_match": float(np.mean(sm_scores)),
        "by_difficulty": {
            f"diff_{d}": float(np.mean(scores)) if scores else 0.0
            for d, scores in by_difficulty.items()
        },
        "by_task_type": {
            k: float(np.mean(v)) for k, v in by_type.items()
        },
        "em_std": float(np.std(em_scores)),
    }

    if uncertainties is not None:
        briers = [brier_score(prob, gt_flat) for prob, gt_flat in uncertainties if prob is not None]
        results["brier_score"] = float(np.mean(briers)) if briers else None

    return results
