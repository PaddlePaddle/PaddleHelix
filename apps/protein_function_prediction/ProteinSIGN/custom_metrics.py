import numpy as np
from sklearn import metrics


def avg_p_r_c(y_true, y_pred, thresholds):
    y_true = y_true[None]
    y_pred = y_pred[None]
    mtx = thresholds[:, None, None] <= (y_pred)

    pred_as_true = mtx.sum(-1)
    real_true = y_true.sum(-1)

    tp = ((y_true == 1) & mtx).sum(-1)
    deno = pred_as_true + 1e-10
    # deno[deno == 0] = 1
    p = tp / deno

    deno = real_true + 1e-10
    # deno[deno == 0] = 1
    r = tp / deno

    n_above_thresholds = np.any(mtx, axis=-1).sum(-1)

    deno = n_above_thresholds + 1e-10
    # deno[deno == 0] = 1
    precisions = p.sum(-1) / deno

    deno = y_true.shape[1]
    recalls = r.sum(-1) / deno

    assert precisions.shape == thresholds.shape
    assert recalls.shape == thresholds.shape

    return precisions, recalls, thresholds


def compute_aupr(true_labels, pred_scores):
    available_index = true_labels.sum(0).astype("bool")
    true_labels = true_labels[:, available_index]
    pred_scores = pred_scores[:, available_index]

    aupr = metrics.average_precision_score(true_labels, pred_scores, average="macro")

    return aupr


def do_compute_metrics(true_labels, pred_scores):

    available_index = true_labels.sum(0).astype("bool")
    true_labels = true_labels[:, available_index]
    pred_scores = pred_scores[:, available_index]

    ########## ******************protein-centric F_max*****************===============
    precision = dict()
    recall = dict()
    ap = dict()
    thresholds = dict()
    max_thrs = dict()

    prot_available_index = true_labels.sum(1).astype("bool")
    prot_true_labels = true_labels[prot_available_index]
    prot_pred_scores = pred_scores[prot_available_index]
    for i in range(int(prot_true_labels.shape[0])):
        precision[i], recall[i], thresholds[i] = metrics.precision_recall_curve(
            prot_true_labels[i], prot_pred_scores[i]
        )
        max_thrs[i] = thresholds[i].max()
        ap[i] = metrics.average_precision_score(
            prot_true_labels[i], prot_pred_scores[i]
        )
    all_thresholds = np.linspace(0, max(max_thrs.values()), 1000)
    # print('Max value is ', max(max_thrs.values()), end='\t')
    # print('Min value is ', min(max_thrs.values()))

    precision["macro"], recall["macro"], _ = avg_p_r_c(
        true_labels, pred_scores, all_thresholds
    )
    deno = (precision["macro"] + recall["macro"]) + 1e-10
    # deno[deno == 0] = 1
    f1_scores = 2 * (precision["macro"] * recall["macro"]) / deno
    f1_max_indx = np.argmax(f1_scores)
    f1_max = f1_scores.max()

    ########## ******************term-centric AUPRC*****************===============
    auprc = compute_aupr(true_labels, pred_scores)

    return f1_max, auprc
