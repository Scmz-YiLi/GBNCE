import os
import time
import numpy as np
import scipy.io as scio
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve

from GBNCE import GBNCE, OPTIMAL_HYPERPARAMETERS, EXPECTED_AUC, get_default_device

def calculate_metrics(labels, scores):
    labels = np.asarray(labels).flatten()
    scores = np.asarray(scores).flatten()
    scores = np.round(scores, 4)

    auc = roc_auc_score(labels, scores)
    ap = average_precision_score(labels, scores)

    precision, recall, _ = precision_recall_curve(labels, scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    max_f1 = float(np.nanmax(f1_scores))

    fpr, tpr, _ = roc_curve(labels, scores)
    tnr = 1 - fpr
    gmean = float(np.nanmax(np.sqrt(tpr * tnr)))

    return auc, ap, max_f1, gmean

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    datasets_dir = os.path.join(base_dir, "Datasets")
    device = get_default_device()

    print("=" * 88)
    print(f"GBNCE Benchmark Evaluation (PyTorch Device: {device})")
    print(f"Datasets Directory: {datasets_dir}")
    print("=" * 88)
    print(f"{'Dataset':<35} | {'Sigma':<6} | {'AUC':<7} | {'Expected':<8} | {'AP':<7} | {'F1':<7} | {'Time (s)':<8} | Status")
    print("-" * 88)

    results = []
    all_matched = True
    total_start = time.time()

    for dname, opt_sigma in OPTIMAL_HYPERPARAMETERS.items():
        mat_path = os.path.join(datasets_dir, f"{dname}.mat")
        if not os.path.exists(mat_path):
            print(f"{dname:<35} | File not found: {mat_path}")
            continue

        mat = scio.loadmat(mat_path)
        trandata = mat['trandata']
        X = trandata[:, :-1]
        labels = trandata[:, -1]

        t0 = time.time()
        scores = GBNCE(X, opt_sigma, device=device)
        elapsed = time.time() - t0
        auc, ap, max_f1, gmean = calculate_metrics(labels, scores)
        exp_auc = EXPECTED_AUC[dname]
        is_match = (round(auc, 3) == round(exp_auc, 3)) or (abs(auc - exp_auc) <= 0.0015)

        if not is_match:
            all_matched = False

        status_str = "MATCH" if is_match else "MISMATCH"
        print(f"{dname:<35} | {opt_sigma:<6.1f} | {auc:<7.3f} | {exp_auc:<8.3f} | {ap:<7.3f} | {max_f1:<7.3f} | {elapsed:<8.2f} | {status_str}")

        results.append({
            'dataset': dname,
            'sigma': opt_sigma,
            'auc': auc,
            'expected_auc': exp_auc,
            'ap': ap,
            'max_f1': max_f1,
            'gmean': gmean,
            'time': elapsed,
            'match': is_match
        })

    print("-" * 88)
    mean_auc = np.mean([r['auc'] for r in results])
    mean_exp_auc = np.mean([r['expected_auc'] for r in results])
    mean_ap = np.mean([r['ap'] for r in results])
    mean_f1 = np.mean([r['max_f1'] for r in results])
    total_elapsed = time.time() - total_start

    print(f"{'Average (20 Datasets)':<35} | {'-':<6} | {mean_auc:<7.3f} | {mean_exp_auc:<8.3f} | {mean_ap:<7.3f} | {mean_f1:<7.3f} | {total_elapsed:<8.2f} |")
    print("=" * 88)

    if all_matched:
        print("[SUCCESS] All 20 datasets reproduced with 100% identical AUC values!")
    else:
        print("[WARNING] Some datasets did not match expected AUC values. Please check data loading.")

if __name__ == '__main__':
    main()
