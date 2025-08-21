import os, cv2, glob, math, numpy as np
import matplotlib.pyplot as plt
# from scipy.ndimage import gaussian_filter
# import matplotlib.image as mpimg
import clean
import ridge_orientation
# import imageio.v3 as iio
# from PIL import Image
# from scipy.spatial import cKDTree
# from skimage.morphology import (
#     skeletonize, remove_small_objects, disk, closing, dilation
# )
from scipy.ndimage import rotate
# from crossing_number import calculate_minutiae, draw_minutiae
from poincare import find_singularities, draw_singularities
from create_graphs import create_graph
# from augment import rotate
import random
import graph
import fingerprint_feature_extractor
import upload

# --- NEW: for interactivity ---
from matplotlib.widgets import Slider

# optional but helps if available
try:
    from scipy.stats import gaussian_kde
    _HAS_KDE = True
except Exception:
    _HAS_KDE = False

from scipy.ndimage import gaussian_filter1d


def _smooth_pdf(data, xs):
    data = np.asarray(data, dtype=float)
    data = np.clip(data, 0, 1)
    if len(data) == 0:
        return np.zeros_like(xs)
    # KDE if possible; otherwise smoothed histogram fallback
    if _HAS_KDE and len(np.unique(data)) > 1 and len(data) >= 3:
        kde = gaussian_kde(data)                    # automatic bandwidth
        ys = kde(xs)
    else:
        hist, edges = np.histogram(data, bins=40, range=(0, 1), density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        ys = np.interp(xs, centers, hist)
        ys = gaussian_filter1d(ys, sigma=1.2)
    return ys


# --- Metrics helpers (as before) ---
def _roc_pr_curves(pos, neg):
    """Compute ROC and PR points without external deps."""
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    P, N = len(pos), len(neg)
    if P == 0 or N == 0:
        fpr = np.array([0.0, 1.0]); tpr = np.array([0.0, 1.0])
        rec = np.array([0.0, 1.0]); prec = np.array([1.0, 1.0])
        thr = np.array([1.0, 0.0])
        return fpr, tpr, rec, prec, thr

    scores = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones(P, dtype=int), np.zeros(N, dtype=int)])
    order = np.argsort(scores)[::-1]  # sort desc by score
    scores_sorted = scores[order]
    labels_sorted = labels[order]

    tp = np.cumsum(labels_sorted == 1).astype(float)
    fp = np.cumsum(labels_sorted == 0).astype(float)

    uniq_idx = np.r_[np.nonzero(np.diff(scores_sorted))[0], len(scores_sorted)-1]
    tp_u = tp[uniq_idx]; fp_u = fp[uniq_idx]; thr_u = scores_sorted[uniq_idx]

    tpr = tp_u / P
    fpr = fp_u / N
    rec = tpr
    denom = (tp_u + fp_u); denom[denom == 0] = 1.0
    prec = tp_u / denom

    # Pad ends for nice curves
    fpr = np.r_[0.0, fpr, 1.0]
    tpr = np.r_[0.0, tpr, 1.0]
    rec  = np.r_[0.0, rec, 1.0]
    base_prec = P / (P + N)
    prec = np.r_[1.0, prec, base_prec]
    thr  = np.r_[thr_u[0] + 1e-12, thr_u, thr_u[-1] - 1e-12]
    return fpr, tpr, rec, prec, thr


def _auc_trapz(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    order = np.argsort(x)
    return float(np.trapz(y[order], x[order]))


def roc_auc(pos, neg):
    fpr, tpr, _, _, _ = _roc_pr_curves(pos, neg)
    return _auc_trapz(fpr, tpr)


def pr_auc(pos, neg):
    _, _, rec, prec, _ = _roc_pr_curves(pos, neg)
    return _auc_trapz(rec, prec)


def tar_at_far(pos, neg, fars=(0.10, 0.01, 0.001)):
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    out = []
    for far in fars:
        if len(neg):
            t = float(np.quantile(neg, 1.0 - far))
            far_emp = float((neg >= t).mean())
        else:
            t = 1.0; far_emp = float('nan')
        tar = float((pos >= t).mean()) if len(pos) else float('nan')
        out.append({"FAR": far, "threshold": t, "TAR": tar, "FAR_emp": far_emp})
    return out


def eer_from_curves(pos, neg):
    fpr, tpr, _, _, thr = _roc_pr_curves(pos, neg)
    fnr = 1.0 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    return {"EER": eer, "threshold": float(thr[idx])}


# --- NEW: partial AUCs so AUROC/AUPRC change with the slider ---
def partial_normalized_auc_from_threshold(pos, neg, t):
    """
    AUROC/AUPRC computed over all thresholds >= t (i.e., stricter or equal),
    normalized to the span of the selected x-range so values stay in [0,1].
    """
    fpr, tpr, rec, prec, thr = _roc_pr_curves(pos, neg)
    mask = thr >= t
    def _norm_auc(x, y):
        if np.count_nonzero(mask) < 2:
            return float('nan')
        xs, ys = x[mask], y[mask]
        area = _auc_trapz(xs, ys)
        span = float(xs.max() - xs.min())
        return (area / span) if span > 0 else float('nan')
    return _norm_auc(fpr, tpr), _norm_auc(rec, prec)


# --- INTERACTIVE PLOT ---
def plot_similarity_confusion_interactive(twin_score, test_similarity_scores):
    """
    twin_score: dict[file_path] -> (angle, similarity_probability)  (blue, 'positive')
    test_similarity_scores: list[similarity_probability]            (red,  'negative')
    """
    # Extract arrays and clip
    pos = np.array([sp for (_, sp) in twin_score.values()], dtype=float)
    neg = np.array(test_similarity_scores, dtype=float)
    pos = np.clip(pos, 0, 1); neg = np.clip(neg, 0, 1)

    # x-grid and smooth PDFs (fixed, not recomputed on slider)
    xs = np.linspace(0, 1, 600)
    y_pos = _smooth_pdf(pos, xs)
    y_neg = _smooth_pdf(neg, xs)

    # Default threshold at 90th percentile of negatives
    B0 = float(np.quantile(neg, 0.90)) if len(neg) else 0.9

    # Precompute full-curve AUCs (for reference if you want)
    # full_auroc = roc_auc(pos, neg)
    # full_auprc = pr_auc(pos, neg)

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    plt.subplots_adjust(bottom=0.16)  # make room for the slider

    def _draw(thr_val):
        ax.clear()

        # Base distributions
        ax.plot(xs, y_neg, color='red', lw=2, label='Unrelated Pairs')
        ax.fill_between(xs, 0, y_neg, color='red', alpha=0.15)
        ax.plot(xs, y_pos, color='blue', lw=2, label='Twin Pairs')
        ax.fill_between(xs, 0, y_pos, color='blue', alpha=0.15)

        # Rugs
        if len(neg): ax.plot(neg, np.zeros_like(neg)-0.002, 'o', ms=3, alpha=0.65, color='red', label='_nolegend_')
        if len(pos): ax.plot(pos, np.zeros_like(pos)+0.002, 'o', ms=3, alpha=0.65, color='blue', label='_nolegend_')

        # Vertical threshold
        ymax = max((y_pos.max() if len(pos) else 1.0), (y_neg.max() if len(neg) else 1.0)) * 1.06
        ax.axvline(thr_val, ls='--', color='k', alpha=0.7)
        ax.text(thr_val, ymax*0.985, 'Threshold', ha='center', va='top', fontsize=11)

        # Shade confusion regions
        m = xs <= thr_val
        ax.fill_between(xs[m], 0, y_neg[m], color='red', alpha=0.30, label='True Negatives')   # TN
        ax.fill_between(xs[m], 0, y_pos[m], color='#e319b4', alpha=0.30, label='False Negatives')  # FN
        m = xs > thr_val
        ax.fill_between(xs[m], 0, y_neg[m], color="#8025cf", alpha=0.30, label='False Positives')  # FP
        ax.fill_between(xs[m], 0, y_pos[m], color='blue', alpha=0.30, label='True Positives')   # TP

        # Axes/legend
        ax.set_xlim(0, 1)
        ax.set_ylim(0, ymax)
        ax.set_xlabel("Similarity Score", fontsize=11)
        ax.set_ylabel("Probability Density", fontsize=11)
        ax.set_title("Weisfeiler–Lehman Graph Isomorphism", fontsize=13)
        handles, labels = ax.get_legend_handles_labels()
        uniq = dict(zip(labels, handles))
        ax.legend(uniq.values(), uniq.keys(), frameon=False, ncol=2, loc='upper center')

        # Current-threshold metrics
        P = len(pos); N = len(neg)
        tar = float((pos >= thr_val).mean()) if P else float('nan')  # True Accept Rate (TPR)
        far = float((neg >= thr_val).mean()) if N else float('nan')  # False Accept Rate (FPR)

        # Partial/segment AUCs so they change with slider
        p_auroc, p_auprc = partial_normalized_auc_from_threshold(pos, neg, thr_val)

        # Bold labels (not numbers) using mathtext
        lines = [
            r"$\mathbf{False\ Accept\ Rate}$: " + (f"{far*100:.2f}%" if not math.isnan(far) else "NA"),
            r"$\mathbf{True\ Accept\ Rate}$: "  + (f"{tar*100:.2f}%" if not math.isnan(tar) else "NA"),
            r"$\mathbf{AUROC}$: "               + (f"{p_auroc:.3f}" if not math.isnan(p_auroc) else "NA"),
            r"$\mathbf{AUPRC}$: "               + (f"{p_auprc:.3f}" if not math.isnan(p_auprc) else "NA"),
        ]
        metrics_text = "\n".join(lines)
        ax.text(0.012, ymax*0.94, metrics_text, ha='left', va='top', fontsize=11,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.7", alpha=0.9))

    # Initial draw
    _draw(B0)

    # Slider
    ax_slider = plt.axes([0.15, 0.005, 0.7, 0.035])  # [left, bottom, width, height] in figure coords
    thr_slider = Slider(ax=ax_slider, label='Threshold', valmin=0.0, valmax=1.0, valinit=B0, valstep=0.001)

    def _on_change(val):
        _draw(thr_slider.val)
        fig.canvas.draw_idle()

    thr_slider.on_changed(_on_change)
    plt.tight_layout()
    plt.show()


# --- Keep your tuple plot helper unchanged ---
def plot_tuples(data):
    """
    Sort a list of (x, y) tuples by x and plot them.
    """
    data = sorted(data, key=lambda t: t[0])
    xs, ys = zip(*data)
    plt.figure(figsize=(6, 4))
    plt.plot(xs, ys, marker='o')
    plt.xlabel("X"); plt.ylabel("Y"); plt.title("Tuples plotted after sorting by X")
    plt.grid(True); plt.tight_layout(); plt.show()


if __name__ == "__main__":
    ROOT_DIR  = "fingerprints"
    # SUBDIRS   = ["DB1_B", "DB2_B", "DB3_B", "DB4_B"]
    SUBDIRS   = ["DB3_B"]  # db4, db3, db2, db1
    graphs = []
    rotated_graphs = []
    twin_score = {}
    test_similarity_scores = []
    count = 0
    correct = 0
    n = 30
    angle_scores = []

    for db in SUBDIRS:
        dir_path = os.path.join(ROOT_DIR, db)
        if not os.path.isdir(dir_path):
            print(f"⚠️  {dir_path} not found – skipping")
            continue

        for root, _, files in os.walk(dir_path):
            for fname in files:
                _, ext = os.path.splitext(fname)
                file_path = os.path.join(root, fname)

                g, rot_g, angle = upload.upload(file_path)

                # twin_score is a dict with key=file_path, value=(angle, similarity_probability)
                similarity_probability, match = graph.wl_graph_similarity(rot_g, g, 1)
                twin_score[file_path] = (angle, similarity_probability)

                graphs.append((file_path, g))
                rotated_graphs.append((file_path, rot_g))


    # Build unrelated pairs list
    for i in range(len(graphs)):
        for j in range(i, len(rotated_graphs)):
            name, g = graphs[i]
            rot_name, rot_g = rotated_graphs[j]
            if name == rot_name:
                continue
            similarity_probability_1, match_1 = graph.wl_graph_similarity(rot_g, g, 1)
            test_similarity_scores.append(similarity_probability_1)

    # --- NEW: interactive plot with slider ---
    plot_similarity_confusion_interactive(twin_score, test_similarity_scores)
