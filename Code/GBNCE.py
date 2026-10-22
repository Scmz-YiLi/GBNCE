import os
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from GB_generation import get_GB

OPTIMAL_HYPERPARAMETERS = {
    'chess_nowin_16_variant1': 1.0,
    'monks_0_12_variant1': 0.6,
    'vote_republican_29_variant1': 0.7,
    'cardio': 1.2,
    'cardiotocography_2and3_33_variant1': 0.7,
    'ecoli': 0.7,
    'ionosphere_b_24_variant1': 1.0,
    'iris_Irisvirginica_11_variant1': 0.5,
    'musk': 0.8,
    'pageblocks_1_258_variant1': 1.4,
    'pendigits': 1.9,
    'vertebral': 0.2,
    'waveform_0_100_variant1': 1.0,
    'yeast_ERL_5_variant1': 0.2,
    'annealing_variant1': 0.9,
    'bands_band_34_variant1': 1.3,
    'bands_band_42_variant1': 1.4,
    'hepatitis_2_9_variant1': 2.0,
    'sick_sick_35_variant1': 1.5,
    'thyroid_disease_variant1': 0.2,
}

EXPECTED_AUC = {
    'chess_nowin_16_variant1': 0.972,
    'monks_0_12_variant1': 0.969,
    'vote_republican_29_variant1': 0.992,
    'cardio': 0.965,
    'cardiotocography_2and3_33_variant1': 0.874,
    'ecoli': 0.925,
    'ionosphere_b_24_variant1': 1.000,
    'iris_Irisvirginica_11_variant1': 1.000,
    'musk': 1.000,
    'pageblocks_1_258_variant1': 0.971,
    'pendigits': 0.959,
    'vertebral': 0.655,
    'waveform_0_100_variant1': 0.782,
    'yeast_ERL_5_variant1': 1.000,
    'annealing_variant1': 0.834,
    'bands_band_34_variant1': 0.785,
    'bands_band_42_variant1': 0.792,
    'hepatitis_2_9_variant1': 0.975,
    'sick_sick_35_variant1': 0.880,
    'thyroid_disease_variant1': 0.760,
}

def get_default_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def compute_NR(X_sub, sigma):
    m = X_sub.shape[1]
    radius = torch.std(X_sub) / sigma
    dist = torch.cdist(X_sub, X_sub, p=2) / np.sqrt(m)
    return (dist <= radius).float()

def compute_single_NCE(NR, total_pairs):
    sizes = torch.sum(NR, dim=-1)
    indistinguishable_pairs = (sizes * (sizes - 1)) / 2.0
    entropy = (total_pairs - indistinguishable_pairs) / total_pairs
    return torch.mean(entropy)

def compute_fast_leave_one_out(NR, total_pairs_x):
    n = NR.shape[0]
    sizes_full = torch.sum(NR, dim=-1)
    total_sum = torch.sum(NR)
    
    sizes_mat = sizes_full.unsqueeze(1) - NR
    sizes_mat.fill_diagonal_(0.0)
    
    pairs_mat = sizes_mat * (sizes_mat - 1) / 2.0
    sum_pairs_for_i = torch.sum(pairs_mat, dim=0)
    NCE_x = (total_pairs_x - (sum_pairs_for_i / (n - 1))) / total_pairs_x
    
    temp_sum = total_sum - 2 * sizes_full + 1.0
    rnc = sizes_full - (temp_sum / (n - 1))
    
    return NCE_x, rnc, sizes_full

def compute_nod(rne, rnc):
    n = rnc.shape[0]
    abs_rnc = torch.abs(rnc)
    nod_pos = rne * (n - abs_rnc) / (2 * n)
    nod_neg = rne * torch.sqrt((n + abs_rnc) / (2 * n))
    return torch.where(rnc >= 0, nod_pos, nod_neg)

def GBNCE(X, sigma, device=None):
    if device is None:
        device = get_default_device()

    X_processed = X.copy()
    nominal_cols = (X_processed >= 1).all(axis=0) & (X_processed.max(axis=0) != X_processed.min(axis=0))
    if any(nominal_cols):
        scaler = MinMaxScaler()
        X_processed[:, nominal_cols] = scaler.fit_transform(X_processed[:, nominal_cols])

    GBs = get_GB(X_processed)
    n_gb = len(GBs)
    m = X_processed.shape[1]

    centers = np.zeros((n_gb, m))
    for idx, gb in enumerate(GBs):
        centers[idx] = np.mean(gb[:, :-1], axis=0)

    X_torch = torch.from_numpy(centers).float().to(device)
    n = n_gb
    total_pairs = n * (n - 1) / 2.0
    total_pairs_x = (n - 1) * (n - 2) / 2.0

    Imp = torch.zeros(m, device=device)
    for k in range(m):
        nr_k = compute_NR(X_torch[:, [k]], sigma)
        Imp[k] = compute_single_NCE(nr_k, total_pairs)

    b_as = torch.argsort(Imp)
    b_de = torch.flip(b_as, dims=[0])

    def process_sequence(attr_order):
        NCE = torch.zeros(m, device=device)
        NCE_x = torch.zeros((n, m), device=device)
        rnc = torch.zeros((n, m), device=device)
        sizes = torch.zeros((n, m), device=device)

        for k in range(m):
            sub_idx = attr_order[:m - k]
            NR = compute_NR(X_torch[:, sub_idx], sigma)
            NCE[k] = compute_single_NCE(NR, total_pairs) + 1e-4
            NCE_x[:, k], rnc[:, k], sizes[:, k] = compute_fast_leave_one_out(NR, total_pairs_x)

        return NCE, NCE_x, rnc, sizes

    NCE_as, NCE_as_x, rnc_as, sizes_as = process_sequence(b_as)
    NCE_de, NCE_de_x, rnc_de, sizes_de = process_sequence(b_de)

    rne_as = torch.clamp(1 - NCE_as_x / NCE_as.unsqueeze(0), 0, 1)
    rne_de = torch.clamp(1 - NCE_de_x / NCE_de.unsqueeze(0), 0, 1)

    nod_as = compute_nod(rne_as, rnc_as)
    nod_de = compute_nod(rne_de, rnc_de)

    w_as = torch.sqrt(sizes_as / n)
    w_de = torch.sqrt(sizes_de / n)

    OF_gb = 1 - (torch.sum((1 - nod_de) * w_de, dim=1) + torch.sum((1 - nod_as) * w_as, dim=1)) / (2 * m)
    OF_np = OF_gb.cpu().numpy()

    sample_scores = np.zeros(len(X))
    for idx, gb in enumerate(GBs):
        point_idxs = gb[:, -1].astype(int)
        sample_scores[point_idxs] = OF_np[idx]

    return np.round(sample_scores, 4)
