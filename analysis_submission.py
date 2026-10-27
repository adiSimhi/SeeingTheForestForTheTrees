import os
import json
import glob
import functools
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from collections import defaultdict

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.titlesize": 20,
})


LABEL_SET = {"A", "B"}



def _model_name(fpath):
    base  = os.path.basename(fpath).replace(".json", "")
    parts = base.split("_vs_")
    if len(parts) >= 2:
        return parts[-2].split("_")[-1]
    return base


def _dataset_name(fpath):
    base = os.path.basename(fpath)
    if "_summary_" in base:
        name = base.split("_summary_")[0].capitalize()
        return "Self-Coordination" if name == "Deception" else name
    return "unknown"


def _is_cross_model(fpath, file_suffix=""):
    """Return True when the two model paths in the filename differ."""
    base = os.path.basename(fpath).replace(".json", "")
    if "_summary_" not in base:
        return False
    rest = base.split("_summary_", 1)[1]
    parts = rest.split("_vs_")
    if len(parts) < 2:
        return False
    m1 = parts[0]
    m2_suf = "_vs_".join(parts[1:])
    if file_suffix:
        tag = "_" + file_suffix
        m2 = m2_suf[:-len(tag)] if m2_suf.endswith(tag) else m2_suf
    else:
        m2 = m2_suf
    return m1 != m2


def _model_file_key(fpath, file_suffix=""):
    """Grouping key: short model name for same-model files, 'M1_vs_M2' for cross-model."""
    base = os.path.basename(fpath).replace(".json", "")
    if "_summary_" not in base:
        return _model_name(fpath)
    rest = base.split("_summary_", 1)[1]
    parts = rest.split("_vs_")
    if len(parts) < 2:
        return _model_name(fpath)
    m1_short = parts[0].split("_")[-1]
    m2_suf = "_vs_".join(parts[1:])
    if file_suffix:
        tag = "_" + file_suffix
        m2_path = m2_suf[:-len(tag)] if m2_suf.endswith(tag) else m2_suf
    else:
        m2_path = m2_suf
    m2_short = m2_path.split("_")[-1]
    if m1_short == m2_short:
        return m1_short
    return "%s_vs_%s" % (m1_short, m2_short)


def _short_model_name(name):
    """
    Friendly display name for figure titles/legends (e.g. "gemma-4-12B-it" -> "Gemma").
    Falls back to the input with any "org/" prefix stripped if no known family matches.
    """
    base = name.split("/")[-1]
    low = base.lower()
    families = ["ministral", "mistral", "gemma", "qwen", "llama",
               "gpt-oss", "gemini", "gpt", "claude"]
    for fam in families:
        if fam in low:
            return "GPT-OSS" if fam == "gpt-oss" else fam.capitalize()
    return base


# Display names used only when a dataset is drawn on a plot (titles, legends).
# The real dataset name stays the dict key/filter value everywhere else.
_DS_DISPLAY_NAMES = {"Self-Coordination": "Self-Coord.", "Behavioral": "Beh-Disp"}


def _short_dataset_name(ds):
    return _DS_DISPLAY_NAMES.get(ds, ds)


def _grid_heatmap(ax, grid, vmin, vmax, cmap):
    """
    Draw a small 2D grid as a heatmap via pcolormesh (vector, not raster) so it
    doesn't render washed-out in a PDF. Tick/text positions must use cell centers (i+0.5).
    """
    return ax.pcolormesh(grid, vmin=vmin, vmax=vmax, cmap=cmap, edgecolors="none")


def _load_model_files(results_dir, file_suffix, skip_models=()):
    """
    file_suffix: a suffix string, or list of suffixes (e.g. runs saved at different
    n_examples), each matched independently. Cross-model files (two different
    models in one conversation) are always skipped -- same-model self-play only.
    """
    suffixes = [file_suffix] if isinstance(file_suffix, str) else list(file_suffix)
    grouped = defaultdict(list)
    seen_files = set()
    for suf in suffixes:
        files = sorted(glob.glob(os.path.join(results_dir, "*.json")))
        if suf:
            files = [f for f in files
                     if os.path.splitext(os.path.basename(f))[0].endswith(suf)]
        for f in files:
            if f in seen_files:
                continue
            seen_files.add(f)
            if _is_cross_model(f, suf):
                continue
            name = _model_file_key(f, suf)
            if any(s in name for s in skip_models):
                continue
            grouped[name].append(f)
    return grouped


@functools.lru_cache(maxsize=None)
def _load_summary(fpath):
    """
    Load a results JSON produced by generate_data_submission.py.
    @lru_cache'd since many analysis functions share the same underlying files.
    """
    with open(fpath) as fh:
        return json.load(fh)


def _disagree_starting(seq1, seq2, T):
    """
    Scenario-level filter (scenario_filter for _collect_conversations*): keeps a
    scenario only if the two speakers' turn-0 labels are valid and DIFFER.
    """
    return seq1[0] in LABEL_SET and seq2[0] in LABEL_SET and seq1[0] != seq2[0]


def _agree_starting(seq1, seq2, T):
    """
    Symmetric to _disagree_starting: turn-0 labels valid and EQUAL. The universal
    source of "agree" examples, since disagree-starting's turn 1 can only be "disagree".
    """
    return seq1[0] in LABEL_SET and seq2[0] in LABEL_SET and seq1[0] == seq2[0]


def _collect_conversations(fpaths, results_dir, inner_key, layer_idx,
                           min_turns=2, scenario_filter=None, tag_ds=False,
                           certainty_key=None, load_swap=False):
    """
    Shared conversation-loading pass: both speakers of every scenario become independent
    entries {'hs', 'seq_s', 'seq_o', 'T', ...}, requiring >= `min_turns` decodable turns.
    """
    conv_list = []
    scen_counter = 0
    for fpath in fpaths:
        ds_name = _dataset_name(fpath) if tag_ds else None
        summary = _load_summary(fpath)

        for scen_key, data in summary.items():
            seq1   = data.get("labels_p1", [])
            seq2   = data.get("labels_p2", [])
            inner1 = data.get(f"{inner_key}_p1", [])
            inner2 = data.get(f"{inner_key}_p2", [])
            cert1  = data.get(f"{certainty_key}_p1", []) if certainty_key else []
            cert2  = data.get(f"{certainty_key}_p2", []) if certainty_key else []
            T = min(len(seq1), len(seq2),
                    len(inner1) if inner1 else 0,
                    len(inner2) if inner2 else 0)
            if T < 2:
                continue
            if scenario_filter is not None and not scenario_filter(seq1, seq2, T):
                continue

            scen_id = scen_counter
            scen_counter += 1
            for seq_s, seq_o, inner_s, cert_s in [(seq1, seq2, inner1, cert1),
                                                   (seq2, seq1, inner2, cert2)]:
                hs, conf = {}, {}
                for t in range(T):
                    h_raw = inner_s[t]
                    if h_raw is None:
                        continue
                    if isinstance(h_raw[0], list):
                        if layer_idx >= len(h_raw):
                            continue
                        h = np.array(h_raw[layer_idx], dtype=np.float32)
                    else:
                        h = np.array(h_raw, dtype=np.float32)
                    if h.ndim != 1:
                        continue
                    hs[t] = h
                    if certainty_key:
                        c = cert_s[t] if t < len(cert_s) else None
                        n = c.get("n_samples") if c else None
                        conf[t] = (c["agree_with_deterministic"] / n
                                  if c and n else float("nan"))

                if len(hs) < min_turns:
                    continue
                conv = {'hs': hs, 'seq_s': seq_s, 'seq_o': seq_o, 'T': T,
                        'scen_id': scen_id, 'scen_key': scen_key}
                if certainty_key:
                    conv['conf'] = conf
                if tag_ds:
                    conv['ds'] = ds_name
                if load_swap:
                    conv['swap'] = bool(data.get('swap', False))
                conv_list.append(conv)
    return conv_list


def _train_test_split_convs(conv_list, seed=42, train_frac=0.7):
    """
    SCENARIO-level train/test split: groups by 'scen_id' so both speakers
    of one scenario land on the same side (avoids leaking shared context).
    """
    import random as _stdlib_random
    from collections import defaultdict as _defaultdict

    n_conv = len(conv_list)
    groups = _defaultdict(list)
    for i, conv in enumerate(conv_list):
        groups[conv.get('scen_id', i)].append(i)

    group_keys = list(groups.keys())
    _stdlib_random.Random(seed).shuffle(group_keys)
    n_groups = len(group_keys)
    n_split = max(1, min(n_groups - 1, int(round(n_groups * train_frac))))
    train_groups = set(group_keys[:n_split])
    test_groups  = set(group_keys[n_split:])

    train_idx = {i for g in train_groups for i in groups[g]}
    test_idx  = {i for g in test_groups  for i in groups[g]}
    train_conv_list = [conv_list[i] for i in range(n_conv) if i in train_idx]
    test_conv_list  = [conv_list[i] for i in range(n_conv) if i in test_idx]
    return train_idx, test_idx, train_conv_list, test_conv_list



def _mean_se(vals):
    if not vals:
        return np.nan, 0.0
    a = np.array(vals, dtype=float)
    return float(a.mean()), float(a.std() / np.sqrt(len(a))) if len(a) > 1 else 0.0


def _auroc(scores, labels):
    """AUROC via Mann-Whitney U. scores: higher = predicted more likely positive (1)."""
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores)           # ascending: highest score gets highest rank
    sorted_labels = labels[order]
    pos_idx = np.where(sorted_labels == 1)[0]   # 0-based ranks of positives
    U = float(pos_idx.sum()) - n_pos * (n_pos - 1) / 2.0
    return float(U / (n_pos * n_neg))


def _auroc_pooled(vals_a, vals_b):
    """
    Point-estimate AUROC(a vs b) -- no significance test/CI, just the threshold-free
    separability statistic. NaN if either group has fewer than 2 values.
    """
    a = np.asarray(vals_a, dtype=float)
    b = np.asarray(vals_b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = np.concatenate([a, b])
    labels = np.array([1] * len(a) + [0] * len(b))
    return _auroc(pooled, labels)


def _naive_powerlaw(p, alpha, beta, gamma):
    """z = alpha * p**gamma + beta. gamma >= 0 is enforced by the caller's
    least_squares bounds, not here (also used to plot an already-fit curve)."""
    return alpha * np.power(np.clip(p, 0.0, None), gamma) + beta


def _fit_powerlaw(turn_p, turn_z, min_samples=5, n_boot=200, seed=42):
    """
    Fits z ~= alpha * p**gamma + beta via bounded, inverse-SE-weighted nonlinear least
    squares against per-turn-number means, then bags over n_boot resamples (median).
    """
    turn_p = np.asarray(turn_p, dtype=float)
    turn_z = np.asarray(turn_z, dtype=float)


    uniq_p = np.unique(turn_p)
    agg_p, agg_z, agg_se = [], [], []
    for p_val in uniq_p:
        zs = turn_z[turn_p == p_val]
        agg_p.append(p_val)
        agg_z.append(float(zs.mean()))
        agg_se.append(float(zs.std() / np.sqrt(len(zs))) if len(zs) > 1 else float("nan"))
    agg_p = np.array(agg_p); agg_z = np.array(agg_z); agg_se = np.array(agg_se)

    n = len(agg_p)
    if n < min_samples:
        return None

    sigma = np.where(np.isnan(agg_se) | (agg_se < 1e-9), 1.0, agg_se)

    ALPHA_BOUND_MULT = 4.0
    GAMMA_MAX = 10.0

    z_range = float(agg_z.max() - agg_z.min()) or 1.0
    p_width = float(agg_p.max() - agg_p.min())
    alpha_bounds = (0.0, ALPHA_BOUND_MULT * z_range / p_width)
    beta_bounds  = (float(agg_z.min() - 0.5 * z_range), float(agg_z.max() + 0.5 * z_range))
    bounds = ([alpha_bounds[0], beta_bounds[0], 0.0],
             [alpha_bounds[1], beta_bounds[1], GAMMA_MAX])
    p0 = (min(z_range, alpha_bounds[1]) or 1.0,
          float(np.clip(agg_z.min(), *beta_bounds)),
          1.0)

    def _try_fit(p_arr, z_arr, s_arr, p0_):
        def _resid(params):
            alpha, beta, gamma = params
            return (z_arr - _naive_powerlaw(p_arr, alpha, beta, gamma)) / s_arr
        try:
            res = least_squares(_resid, x0=p0_, bounds=bounds, max_nfev=20000)
            return res.x if res.success else None
        except ValueError:
            return None

    base_popt = _try_fit(agg_p, agg_z, sigma, p0)
    if base_popt is None:
        return None

    # ── bootstrap-bagging over the aggregated points ─────────────────────
    rng = np.random.default_rng(seed)
    boot_alpha, boot_beta, boot_gamma = [], [], []
    idx_all = np.arange(n)
    for _ in range(n_boot):
        sampled = rng.choice(idx_all, size=n, replace=True)
        popt_b = _try_fit(agg_p[sampled], agg_z[sampled], sigma[sampled], base_popt)
        if popt_b is not None:
            boot_alpha.append(popt_b[0]); boot_beta.append(popt_b[1]); boot_gamma.append(popt_b[2])

    if boot_alpha:
        alpha, beta, gamma = float(np.median(boot_alpha)), float(np.median(boot_beta)), float(np.median(boot_gamma))
    else:
        alpha, beta, gamma = (float(x) for x in base_popt)

    pred = _naive_powerlaw(agg_p, alpha, beta, gamma)
    ss_res = float(np.sum((agg_z - pred) ** 2))
    ss_tot = float(np.sum((agg_z - agg_z.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")

    return {"alpha": alpha, "beta": beta, "gamma": gamma, "r2": r2, "n": n,
            "agg_p": agg_p, "agg_z": agg_z, "agg_se": agg_se}


def _balance_by_turn(feat_train, turn_train, seed=0):
    """
    Indices making the +1/-1 classes in feat_train exactly equal in count AT EVERY
    TURN, so the fit can't use turn position as a label shortcut.
    """
    feat_train = np.asarray(feat_train)
    turn_train = np.asarray(turn_train)
    rng = np.random.default_rng(seed)
    idx_pos, idx_neg = [], []
    for t in np.unique(turn_train):
        at_t = np.where(turn_train == t)[0]
        pos_idx = at_t[feat_train[at_t] > 0]
        neg_idx = at_t[feat_train[at_t] < 0]
        n_t = min(len(pos_idx), len(neg_idx))
        if n_t < 1:
            continue
        idx_pos.extend(rng.choice(pos_idx, size=n_t, replace=False))
        idx_neg.extend(rng.choice(neg_idx, size=n_t, replace=False))
    return np.array(idx_pos + idx_neg, dtype=int)


def _turn_reweight(turn_vals):
    """
    Per-example weight = 1/(count of that turn in `turn_vals`), rescaled to sum to
    len(turn_vals) -- makes every turn contribute equal total weight, without discarding data.
    """
    turn_vals = np.asarray(turn_vals)
    uniq, counts = np.unique(turn_vals, return_counts=True)
    count_by_turn = dict(zip(uniq.tolist(), counts.tolist()))
    w = np.array([1.0 / count_by_turn[t] for t in turn_vals.tolist()])
    w *= len(turn_vals) / w.sum()
    return w


def _ridge_lda_dir(R_c, feat_vals, reg_grid=None, sample_weight=None):
    """
    Regularized (ridge) discriminant direction -- Fisher's LDA analogue, solved via the
    n x n dual/kernel form (Woodbury), lambda by LOOCV. Returns (u, lambda) or None.
    """
    f = np.asarray(feat_vals, dtype=np.float64)
    f_c = f - f.mean()
    s = f_c.std()
    if s < 1e-10:
        return None
    f_c = f_c / s

    R_c64 = R_c.astype(np.float64)
    if sample_weight is not None:
        sw = np.sqrt(np.asarray(sample_weight, dtype=np.float64))
        R_c64 = R_c64 * sw[:, None]
        f_c = f_c * sw
    K = R_c64 @ R_c64.T                      # (n, n) Gram matrix
    eigvals, V = np.linalg.eigh(K)           # K = V diag(eigvals) V^T
    eigvals = np.clip(eigvals, 0.0, None)
    Vf = V.T @ f_c

    if reg_grid is None:
        pos = eigvals[eigvals > 1e-8]
        scale = float(pos.mean()) if len(pos) else 1.0
        reg_grid = scale * np.array([1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0])

    best_lam, best_err = float(reg_grid[0]), np.inf
    for lam in reg_grid:
        d_i  = eigvals / (eigvals + lam)               # diag of D in K = V D V^T solve
        yhat = V @ (d_i * Vf)                           # fitted values for this λ
        h_ii = np.clip((V ** 2) @ d_i, 0.0, 1 - 1e-6)    # leverage (hat-matrix diagonal)
        loo_resid = (f_c - yhat) / (1 - h_ii)
        err = float(np.mean(loo_resid ** 2))
        if err < best_err:
            best_err, best_lam = err, float(lam)

    alpha = V @ (Vf / (eigvals + best_lam))
    w = R_c64.T @ alpha
    nrm = np.linalg.norm(w)
    if nrm < 1e-10:
        return None
    u = (w / nrm).astype(np.float32)
    if np.corrcoef(R_c @ u, f)[0, 1] < 0:
        u = -u
    return u, best_lam


def _logreg_dir(R_c, feat_vals, reg_grid=None, sample_weight=None, n_folds=5, cv_seed=0):
    """
    L2-regularized logistic regression direction -- a discriminative alternative to
    _ridge_lda_dir (same calling convention), via scikit-learn's LogisticRegressionCV.
    """
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.model_selection import StratifiedKFold

    f = np.asarray(feat_vals, dtype=np.float64)
    y = (f > 0).astype(np.float64)
    n = R_c.shape[0]
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    if len(idx_pos) < 2 or len(idx_neg) < 2:
        return None
    R64 = R_c.astype(np.float64)

    if reg_grid is None:
        avg_sq_norm = float(np.mean(np.sum(R64 ** 2, axis=1)))
        scale = n / max(avg_sq_norm, 1e-8)
        reg_grid = scale * np.array([1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0, 30.0, 100.0])
    Cs = 1.0 / np.asarray(reg_grid, dtype=np.float64)

    k = max(2, min(n_folds, len(idx_pos), len(idx_neg)))
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=cv_seed)
    clf = LogisticRegressionCV(Cs=Cs, cv=cv, penalty="l2", solver="lbfgs",
                               scoring="neg_log_loss", fit_intercept=True,
                               max_iter=1000)
    try:
        clf.fit(R64, y, sample_weight=sample_weight)
    except ValueError:
        return None

    w = clf.coef_[0]
    nrm = np.linalg.norm(w)
    if nrm < 1e-10:
        return None
    u = (w / nrm).astype(np.float32)
    if np.corrcoef(R_c @ u, f)[0, 1] < 0:
        u = -u
    best_lam = float(1.0 / clf.C_[0])
    return u, best_lam


# ─── Markov chain validation ────────────────────────────────────────────────

def _run_markov(train, test):
    """
    Fits a first-order Markov chain (plus a unigram and a second-order baseline) on
    `train`, scored on `test` -- just the curves plot_markov_2nd_order_baseline
    plots (emp_d/pred_d/pred_d_2nd/pred_d_uni/max_T), no summary statistics.

    States are 1 = agree, 0 = disagree, so every returned curve is an AGREE
    share (the emp_d/pred_d/pi_dis names are kept from the earlier
    disagree-oriented version).
    """
    if len(train) < 3 or not test:
        return None

    # fit transition matrix
    counts = np.zeros((2, 2))
    for seq in train:
        for t in range(len(seq) - 1):
            counts[int(seq[t]), int(seq[t + 1])] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    T_mat = counts / row_sums

    # SECOND-ORDER Markov chain (context = last TWO states) fit on the same
    # TRAIN split, as a 4-state chain over ordered pairs idx=2*prev+cur.
    counts2 = np.zeros((4, 4))
    for seq in train:
        for t in range(1, len(seq) - 1):
            prev, cur, nxt = int(seq[t - 1]), int(seq[t]), int(seq[t + 1])
            counts2[2 * prev + cur, 2 * cur + nxt] += 1
    row_sums2 = counts2.sum(axis=1, keepdims=True)
    row_sums2[row_sums2 == 0] = 1
    T2_mat = counts2 / row_sums2

    # UNIGRAM (order-0) baseline: P(agree at t) is a single constant,
    # the pooled marginal agree rate over every TRAIN (conversation, turn) pair.
    train_vals = np.concatenate([np.asarray(s, dtype=float) for s in train]) \
        if train else np.array([])
    # train_vals is already 1=agree -- no inversion needed.
    p_uni = float(train_vals.mean()) if len(train_vals) else float("nan")

    # a=P(agree->disagree), b=P(disagree->agree) -- with 1=agree/0=disagree,
    # that's T_mat[1,0]/T_mat[0,1].
    a = float(T_mat[1, 0])
    b = float(T_mat[0, 1])
    lam2   = 1.0 - a - b
    # stationary AGREE probability = b/(a+b) (the complement, and swapped
    # numerator, of the disagree-oriented a/(a+b)).
    pi_dis = b / (a + b) if (a + b) > 1e-12 else 0.5

    # predict a(t) on test. a(t)=pi_agree+(a0-pi_agree)*lam2**t is AFFINE in a0, so
    # averaging a0 first then applying the closed form once == averaging per-conversation.
    max_T = max(len(s) for s in test)
    emp_d  = np.full(max_T, np.nan)
    pred_d = np.full(max_T, np.nan)
    for t in range(max_T):
        in_range = [s for s in test if t < len(s)]
        if not in_range:
            continue
        # in_range's states are already 1=agree -- no inversion needed.
        emp_d[t] = float(np.mean([s[t] for s in in_range]))
        d0_bar   = float(np.mean([s[0] for s in in_range]))
        pred_d[t] = pi_dis + (d0_bar - pi_dis) * (lam2 ** t)

    # second-order prediction: propagates the exact pair-state distribution via
    # repeated T2_mat multiplication.
    pred_d_2nd = np.full(max_T, np.nan)
    for t in range(2, max_T):
        in_range2 = [s for s in test if t < len(s)]
        if not in_range2:
            continue
        dist = np.zeros(4)
        for s in in_range2:
            dist[2 * int(s[0]) + int(s[1])] += 1.0
        dist /= dist.sum()
        for _ in range(t - 1):
            dist = dist @ T2_mat
        # cur=agree pair-states are idx 1/3 under 1=agree/0=disagree.
        pred_d_2nd[t] = float(dist[1] + dist[3])

    pred_d_uni = np.full(max_T, p_uni)

    return dict(T_mat=T_mat, a=a, b=b, lam2=lam2, pi_dis=pi_dis,
                emp_d=emp_d, pred_d=pred_d, max_T=max_T,
                T2_mat=T2_mat, pred_d_2nd=pred_d_2nd,
                p_uni=p_uni, pred_d_uni=pred_d_uni,
                n_train=len(train), n_test=len(test))


def markov_validation(results_dir="results", file_suffix="False_False_False",
                      skip_models=(), seed=42, train_frac=0.7):
    """
    Fits a first-order Markov chain (agree/disagree state per turn) per (model, dataset),
    restricted to turn-0-disagreeing scenarios; plotting left to the caller.
    """
    model_files = _load_model_files(results_dir, file_suffix, skip_models)
    if not model_files:
        return {}

    # ── collect sequences per (model, dataset) ────────────────────────────────
    all_results = {}   # {model: {ds: result_dict}}

    for model_name, mfiles in sorted(model_files.items()):
        seqs_by_ds = defaultdict(list)
        scen_counter = 0
        for fpath in mfiles:
            ds = _dataset_name(fpath)
            summary = _load_summary(fpath)
            for scen_key, data in summary.items():
                labels_p1 = data.get("labels_p1", [])
                labels_p2 = data.get("labels_p2", [])
                T_raw = min(len(labels_p1), len(labels_p2))
                # Longest turn-aligned prefix where BOTH speakers have a parseable
                # label -- stops for good at the first turn where either doesn't.
                T = 0
                while T < T_raw and labels_p1[T] in LABEL_SET and labels_p2[T] in LABEL_SET:
                    T += 1
                if T < 3:
                    continue
                if labels_p1[0] == labels_p2[0]:
                    continue
                # 1 = agree, 0 = disagree -- see this function's docstring.
                state = np.array([labels_p1[t] == labels_p2[t] for t in range(T)], dtype=float)
                seqs_by_ds[ds].append(
                    {'state': state, 'scen_id': scen_counter, 'scen_key': scen_key})
                scen_counter += 1

        all_results[model_name] = {}
        for ds_name, seqs in sorted(seqs_by_ds.items()):
            _, _, train_convs, test_convs = _train_test_split_convs(
                seqs, seed=seed, train_frac=train_frac)
            train = [c['state'] for c in train_convs]
            test  = [c['state'] for c in test_convs]
            res = _run_markov(train, test)
            if res is None:
                continue
            all_results[model_name][ds_name] = res

    return all_results


def plot_markov_2nd_order_baseline(
        all_results, results_dir,
        out_filename="markov_validation_2nd_order.pdf",
        y_label="Disagreement (%)"):
    """
    Spreadsheet-grid plot (rows=model, cols=dataset) of markov_validation's curves
    plus a fourth SECOND-ORDER (last-two-states) Markov curve, from `all_results`.

    y_label lets a caller feed an AGREE-oriented `all_results` (see _run_markov's
    docstring) and label the shared y-axis accordingly, e.g. "Agreement (%)".
    """
    all_models  = sorted(all_results.keys(), key=lambda m: ("gemini" in m.lower(), m))
    all_datasets = sorted({ds for m in all_results for ds in all_results[m]})
    n_rows, n_cols = len(all_models), len(all_datasets)
    if n_rows == 0 or n_cols == 0:
        return None

    has_data = [[all_results[all_models[ri]].get(all_datasets[ci]) is not None
                for ci in range(n_cols)] for ri in range(n_rows)]
    top_row_for_col = [min((ri for ri in range(n_rows) if has_data[ri][ci]), default=None)
                       for ci in range(n_cols)]
    bottom_row_for_col = [max((ri for ri in range(n_rows) if has_data[ri][ci]), default=None)
                          for ci in range(n_cols)]
    left_col_for_row = [min((ci for ci in range(n_cols) if has_data[ri][ci]), default=None)
                        for ri in range(n_rows)]
    # Global (not per-panel) max turn, so every panel shares the same ticks.
    global_max_T = max((r["max_T"] for m in all_results.values() for r in m.values()
                        if r is not None), default=0)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.3 * n_cols, 3.0 * n_rows),
                             squeeze=False, sharex=True, sharey=True)
    # ABSOLUTE inches (not figure-fraction) for fig.text/legend offsets below,
    # so the gap stays fixed regardless of how many models/datasets are in the grid.
    _FIG_H_IN = 3.0 * n_rows
    _FIG_W_IN = 4.3 * n_cols

    for ri, mname in enumerate(all_models):
        for ci, ds in enumerate(all_datasets):
            ax = axes[ri][ci]
            res = all_results[mname].get(ds)
            if res is None:
                ax.set_visible(False); continue

            max_T = res["max_T"]
            turns = np.arange(max_T)
            valid = ~np.isnan(res["emp_d"])
            pred_d_2nd = res.get("pred_d_2nd")
            valid2 = (~np.isnan(pred_d_2nd) if pred_d_2nd is not None
                     else np.zeros(max_T, dtype=bool))

            # Distinct hues (Okabe-Ito colorblind-safe palette) for every
            # curve. Empirical is black (the ground-truth reference every
            # predictor is judged against); Markov second order is blue
            # (#0072B2), freed up by moving empirical off it.
            ax.plot(turns[valid], 100 * res["emp_d"][valid],
                    "o-", color="black", ms=7, lw=4, label="Empirical ground truth")
            ax.plot(turns[valid], 100 * res["pred_d"][valid],
                    "--", color="#D55E00", lw=4, label="Markov (bigram)")
            if valid2.any():
                ax.plot(turns[valid2], 100 * pred_d_2nd[valid2],
                        "-.", color="#0072B2", lw=4, label="Markov second order")
            ax.plot(turns, 100 * res["pred_d_uni"],
                    ":", color="#009E73", lw=4, label="Unigram (no history)")

            ax.set_ylim(0, 100)
            ax.set_yticks([0, 50, 100])
            ax.set_xticks(np.arange(0, global_max_T, 2))
            ax.tick_params(axis='both', labelsize=24,
                           labelbottom=(ri == bottom_row_for_col[ci]),
                           labelleft=(ci == left_col_for_row[ri]))
            if ri == top_row_for_col[ci]:
                ax.set_title(_short_dataset_name(ds), fontsize=22)
            if ci == left_col_for_row[ri]:
                ax.set_ylabel(_short_model_name(mname), fontsize=22,
                             rotation=90, ha="center", va="center", labelpad=10)

    handles = labels = None
    for ax_row in axes:
        for ax in ax_row:
            if ax.get_visible():
                handles, labels = ax.get_legend_handles_labels()
                break
        if handles:
            break
    # Dataset/model labels sit in the title/ylabel slots; actual axis units
    # are stated once, globally, below/left of the whole grid.
    fig.text(0.5, -0.14 / _FIG_H_IN, "Turn Number", ha="center", va="top", fontsize=26)
    fig.text(-0.65 / _FIG_W_IN, 0.5, y_label, ha="right", va="center",
            rotation=90, fontsize=26)
    if handles:
        fig.legend(handles, labels, loc="lower center",
                  bbox_to_anchor=(0.5, -1.2 / _FIG_H_IN),
                  ncol=len(labels), fontsize=24, frameon=True)

    plt.tight_layout()
    out_path = os.path.join(results_dir, out_filename)
    fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)
    return all_results


def _never_flips_label(seq):
    """
    True if `seq` holds exactly ONE A/B opinion for its entire duration -- never
    flips ANYWHERE, not merely "hasn't flipped yet." Used for "golden" reference turns.
    """
    valid = {l for l in seq if l in LABEL_SET}
    return len(valid) <= 1


def estimate_pooled_flip_signal_turn0dir(
        results_dir, file_suffix,
        layer_idx=2, inner_key="inners",
        skip_models=(),
        lags=(-1, 0, 1, 2, 3),
        min_samples=5,
        balanced_max_turn=None,
        seed=42, train_frac=0.7,
        exclude_coflips=False,
        clean_window=2,
        require_single_flip=True,
        direction_feature="agree_disagree",
        turn1_cross_population_dir=False,
        balance=True,
        reweight_across_turns=True,
        classifier="ridge_lda"):
    """
    Studies disagree-starting conversations: after a flip to agree, how does z evolve,
    scored via AUROC against the static "agree & stay" reference pool at every lag
    (including pre-flip lags -- the only reference pool the -1..3 plot ever needs).
    """
    LAGS = sorted(set(lags) | {0})

    model_files = _load_model_files(results_dir, file_suffix, skip_models)
    if not model_files:
        return {}

    all_results = {}

    for model_name, mfiles in sorted(model_files.items()):
        fpaths = sorted(mfiles)
        if not fpaths:
            continue

        # Pass 1: collect per-conversation data; certainty_key="certainty" adds
        # 'conf': {t: agree_with_deterministic/n_samples} for per-lag confidence.
        conv_list = _collect_conversations(fpaths, results_dir, inner_key, layer_idx,
                                           certainty_key="certainty",
                                           scenario_filter=_disagree_starting)
        n_conv = len(conv_list)
        if n_conv < 2:
            continue

        # ── conversation-level train/test split ───────────────────────────
        _, _, train_conv_list, test_conv_list = _train_test_split_convs(
            conv_list, seed=seed, train_frac=train_frac)

        rng_bal = np.random.default_rng(seed)

        # Direction fit: turn-balanced golden unflipped agree/disagree turns via
        # _fit_turn_balanced_direction; "agree" from a SEPARATE agree-starting population.
        agree_conv_list = _collect_conversations(
            fpaths, results_dir, inner_key, layer_idx,
            certainty_key="certainty", scenario_filter=_agree_starting)
        # Same seeded 70/30 split as the disagree side; TRAIN half feeds the fit,
        # TEST half sources the agree&stay reference pool below.
        _, _, agree_train_conv_list, agree_test_conv_list = _train_test_split_convs(
            agree_conv_list, seed=seed, train_frac=train_frac)
        effective_balanced_max_turn = 1 if turn1_cross_population_dir else balanced_max_turn
        fit = _fit_turn_balanced_direction(
            train_conv_list, direction_feature, rng_bal, effective_balanced_max_turn, min_samples,
            agree_conv_list=agree_train_conv_list,
            balance=balance, reweight_across_turns=reweight_across_turns, classifier=classifier)
        if fit is None:
            continue
        u, mu = fit["u"], fit["mu"]

        # Evaluation: project TEST conversations onto (u, mu); build the flip
        # trajectory and the single static GOLDEN "agree & stay" reference pool
        # (never-flips speakers only) -- every displayed lag (-1..3) is scored
        # against this one pool, so no "disagree & stay" pool is ever built.
        samples_by_lag = {lag: [] for lag in LAGS}
        conf_by_lag = {lag: [] for lag in LAGS}
        n_coflips_excluded = 0
        n_isolation_excluded = 0
        n_multiflip_excluded = 0
        # one entry per surviving flip event: {lag: z}, only its available/
        # uncontaminated lags -- lets a caller plot individual trajectories.
        flip_trajectories = []

        for conv in test_conv_list:
            hs, seq_s, seq_o, T = conv['hs'], conv['seq_s'], conv['seq_o'], conv['T']
            z_by_t = {t: float((h - mu) @ u) for t, h in hs.items()}
            conf_by_t = conv.get('conf', {})

            flip_ts_raw = []
            for t in range(1, T):
                if t not in hs:
                    continue
                ls_prev = seq_s[t - 1]; ls_curr = seq_s[t]
                lo_prev = seq_o[t - 1]
                if any(l not in LABEL_SET for l in (ls_prev, ls_curr, lo_prev)):
                    continue
                if lo_prev == ls_prev:
                    # agree-context turn: agree&stay is sourced from
                    # agree_test_conv_list below instead, not from here.
                    continue
                if ls_curr != ls_prev:
                    flip_ts_raw.append(t)

            # Clean flip_ts_raw via the shared rule (co-flip window + isolation
            # against ANY own-label change) -- see _clean_flip_candidates.
            flip_ts, n_coflip_here = _clean_flip_candidates(
                flip_ts_raw, seq_o, T, exclude_coflips=exclude_coflips,
                clean_window=clean_window, seq_s=seq_s)
            n_coflips_excluded += n_coflip_here
            n_isolation_excluded += len(flip_ts_raw) - n_coflip_here - len(flip_ts)

            if require_single_flip and len(flip_ts) != 1:
                n_multiflip_excluded += len(flip_ts)
                continue   # skip this conversation's flip(s) entirely -- see docstring

            # per-lag window contamination is checked against EVERY own-label
            # change (own_change_turns), not just other tracked flip_ts entries.
            own_change_turns = _own_label_change_turns(seq_s, T)
            for t_f in flip_ts:
                event_traj = {}
                for lag in LAGS:
                    target = t_f + lag
                    if target not in z_by_t:
                        continue
                    w_lo, w_hi = min(t_f, target), max(t_f, target)
                    if any(other != t_f and w_lo <= other <= w_hi for other in own_change_turns):
                        continue   # window crosses another own-label change -- contaminated
                    samples_by_lag[lag].append(z_by_t[target])
                    conf_by_lag[lag].append(conf_by_t.get(target, float("nan")))
                    event_traj[lag] = z_by_t[target]
                # flip_trajectories keeps only single-flip conversations.
                if event_traj and len(flip_ts) == 1:
                    flip_trajectories.append(event_traj)

        # agree&stay golden reference pool: agree_test_conv_list's TEST half, same
        # "stop for good" prefix rule as _fit_turn_balanced_direction's _harvest.
        agree_by_turn, agree_conf_by_turn = defaultdict(list), defaultdict(list)
        for conv in agree_test_conv_list:
            hs, seq_s, seq_o, T = conv['hs'], conv['seq_s'], conv['seq_o'], conv['T']
            z_by_t = {t: float((h - mu) @ u) for t, h in hs.items()}
            conf_by_t = conv.get('conf', {})
            for t in range(1, T):
                if t not in hs:
                    continue
                if seq_s[t] not in LABEL_SET:
                    break
                if seq_s[t - 1] not in LABEL_SET or seq_o[t - 1] not in LABEL_SET:
                    break
                if seq_s[t] != seq_s[t - 1]:
                    break   # speaker's own opinion changed -- stop for good
                if seq_s[t] != seq_o[t - 1]:
                    break   # relation left "agree" -- stop for good
                agree_by_turn[t].append(z_by_t[t])
                agree_conf_by_turn[t].append(conf_by_t.get(t, float("nan")))

        rng_agree_ref = np.random.default_rng(seed)
        agree_stay_pool, agree_stay_conf = [], []
        if agree_by_turn:
            n_ref_t = min(len(v) for v in agree_by_turn.values())
            for t in sorted(agree_by_turn):
                idx = rng_agree_ref.choice(len(agree_by_turn[t]), size=n_ref_t, replace=False)
                agree_stay_pool.extend(agree_by_turn[t][i] for i in idx)
                agree_stay_conf.extend(agree_conf_by_turn[t][i] for i in idx)

        astay_mean, astay_se = _mean_se(agree_stay_pool)

        # reference confidence: mean fraction of resampled continuations reproducing
        # the used label, for the static pool (NaNs dropped).
        def _conf_mean(vals):
            v = [x for x in vals if not np.isnan(x)]
            return (float(np.mean(v)), len(v)) if v else (float("nan"), 0)

        astay_conf_mean, astay_conf_n = _conf_mean(agree_stay_conf)

        # ── per-lag test: every lag (including pre-flip) scored against the
        # one static agree&stay pool ─────────────────────────────────────
        lag_list, auroc_list, n_list = [], [], []
        n_flip_list = []
        flip_mean_list, flip_se_list = [], []
        conf_mean_list, conf_n_list = [], []

        for lag in LAGS:
            flips = samples_by_lag[lag]
            n_min = min(len(flips), len(agree_stay_pool))
            lag_list.append(lag); n_list.append(n_min)
            n_flip_list.append(len(flips))

            fm, fs = _mean_se(flips)
            flip_mean_list.append(fm); flip_se_list.append(fs)

            conf_m, conf_n = _conf_mean(conf_by_lag[lag])
            conf_mean_list.append(conf_m); conf_n_list.append(conf_n)

            if n_min < min_samples:
                auroc_list.append(float("nan"))
                continue
            auc = _auroc_pooled(flips, agree_stay_pool)
            auroc_list.append(auc)

        all_results[model_name] = {
            "lags": np.array(lag_list), "auroc": np.array(auroc_list),
            "n": np.array(n_list), "n_flip": np.array(n_flip_list),
            "flip_mean": np.array(flip_mean_list), "flip_se": np.array(flip_se_list),
            "flip_conf_mean": np.array(conf_mean_list), "flip_conf_n": np.array(conf_n_list),
            "agree_stay_mean": astay_mean, "agree_stay_se": astay_se,
            "agree_stay_conf_mean": astay_conf_mean,
            "n_train_total": fit["n_train_total"], "per_turn_n": fit["per_turn_n"],
            "flip_trajectories": flip_trajectories,
            # raw (non-aggregated) golden reference pool, for a caller that
            # needs individual z values, not just the mean+-SE summary above.
            "agree_stay_pool": list(agree_stay_pool),
            # unit-norm fitted direction, for callers that want to inspect it directly.
            "u": fit["u"], "mu": fit["mu"],
        }

    # Plotting is left to the caller.
    return all_results


def _postflip_chance_distance_grid(all_results, lags):
    """
    Shared extraction for the chance-distance-vs-lag view; every lag (r["auroc"]/
    r["n_flip"]) is already scored against the single agree_stay_pool reference --
    see estimate_pooled_flip_signal_turn0dir.
    """
    models = sorted(all_results.keys())
    lags = list(lags)
    n_lags, n_models = len(lags), len(models)

    dist_grid = np.full((n_models, n_lags), np.nan)
    n_flip_grid = np.zeros((n_models, n_lags), dtype=int)

    for ri, mname in enumerate(models):
        r = all_results[mname]
        lag_to_idx = {int(l): i for i, l in enumerate(r["lags"])}
        for ci, lag in enumerate(lags):
            li = lag_to_idx.get(lag)
            if li is None:
                continue
            n_flip_grid[ri, ci] = int(r["n_flip"][li])
            auc = r["auroc"][li]
            if not np.isnan(auc):
                dist_grid[ri, ci] = 2.0 * abs(auc - 0.5)

    return models, dist_grid, n_flip_grid


def plot_chance_distance_postflip_heatmap(
        all_results, results_dir,
        lags=(-1, 0, 1, 2, 3),
        out_filename="pooled_flip_signal_turn0dir_last_nocoflip_heatmap_postflip.pdf"):
    """
    Single-panel chance-distance heatmap (rows=model, cols=lag) from
    estimate_pooled_flip_signal_turn0dir's output (see _postflip_chance_distance_grid).
    """
    lags = list(lags)
    models, dist_grid, _ = _postflip_chance_distance_grid(all_results, lags)
    n_lags, n_models = len(lags), len(models)

    fig, ax = plt.subplots(figsize=(max(6, 1.6 * n_lags), 1.2 + 0.9 * n_models))
    # pcolormesh, not imshow -- see _grid_heatmap's docstring on tiny-grid rasters.
    im = _grid_heatmap(ax, dist_grid, vmin=0.0, vmax=1.0, cmap="Reds")
    ax.invert_yaxis()
    ax.set_xticks(np.arange(n_lags) + 0.5)
    ax.set_xticklabels([("0" if l == 0 else f"+{l}" if l > 0 else str(l)) for l in lags],
                      fontsize=17)
    ax.set_yticks(np.arange(n_models) + 0.5)
    ax.set_yticklabels([_short_model_name(m) for m in models], fontsize=17)

    for ri in range(n_models):
        for ci in range(n_lags):
            v = dist_grid[ri, ci]
            if np.isnan(v):
                ax.text(ci + 0.5, ri + 0.5, "n/a", ha="center", va="center",
                        fontsize=15, color="black")
                continue
            # dark cells render dark -- switch to white text to stay legible.
            ax.text(ci + 0.5, ri + 0.5, f"{v:.2f}", ha="center", va="center",
                    fontsize=15, color="white" if v > 0.35 else "black")

    cbar = fig.colorbar(im, ax=ax, label="chance distance = 2|AUROC-0.5|", fraction=0.03, pad=0.02)
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label("Δℓ", fontsize=17)
    ax.set_xlabel("Lag (Turn Relative to Opinion Change)", fontsize=17)
    plt.tight_layout()
    out = os.path.join(results_dir, out_filename)
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    return dist_grid


def _collect_conversations_with_text(fpaths, results_dir, inner_key, layer_idx, min_turns=2,
                                     scenario_filter=None):
    """
    Like _collect_conversations, but also keeps each scenario's raw "conversation"
    text list, keyed by 'file_key'/'speaker_idx'. Returns (conv_list, text_by_key).
    """
    conv_list = []
    text_by_key = {}
    scen_counter = 0
    for fpath in fpaths:
        ds_name = _dataset_name(fpath)
        summary = _load_summary(fpath)
        for scen_key, data in summary.items():
            seq1   = data.get("labels_p1", [])
            seq2   = data.get("labels_p2", [])
            inner1 = data.get(f"{inner_key}_p1", [])
            inner2 = data.get(f"{inner_key}_p2", [])
            T = min(len(seq1), len(seq2),
                    len(inner1) if inner1 else 0,
                    len(inner2) if inner2 else 0)
            if T < 2:
                continue
            if scenario_filter is not None and not scenario_filter(seq1, seq2, T):
                continue

            scen_id = scen_counter
            scen_counter += 1
            file_key = (fpath, scen_key)
            text_by_key[file_key] = {"conversation": data.get("conversation", []),
                                     "ds": ds_name, "scen_key": scen_key}

            for speaker_idx, (seq_s, seq_o, inner_s) in enumerate(
                    [(seq1, seq2, inner1), (seq2, seq1, inner2)]):
                hs = {}
                for t in range(T):
                    h_raw = inner_s[t]
                    if h_raw is None:
                        continue
                    if isinstance(h_raw[0], list):
                        if layer_idx >= len(h_raw):
                            continue
                        h = np.array(h_raw[layer_idx], dtype=np.float32)
                    else:
                        h = np.array(h_raw, dtype=np.float32)
                    if h.ndim != 1:
                        continue
                    hs[t] = h

                if len(hs) < min_turns:
                    continue
                conv_list.append({'hs': hs, 'seq_s': seq_s, 'seq_o': seq_o, 'T': T,
                                  'scen_id': scen_id, 'scen_key': scen_key,
                                  'file_key': file_key, 'speaker_idx': speaker_idx,
                                  'ds': ds_name})
    return conv_list, text_by_key


def _own_label_change_turns(seq_s, T):
    """
    Every turn where THIS speaker's own label changes, any direction -- the full
    "did anything move here" set, used as _clean_flip_candidates's isolation reference.
    """
    return [t for t in range(1, T)
           if t < len(seq_s) and seq_s[t] in LABEL_SET
           and seq_s[t - 1] in LABEL_SET and seq_s[t] != seq_s[t - 1]]


def _clean_flip_candidates(flip_ts_raw, seq_o, T, exclude_coflips=True, clean_window=2,
                           seq_s=None):
    """
    Drops contaminated flip candidates: exclude_coflips drops one within `clean_window`
    turns of the OTHER speaker's own change; isolation drops one near another own change.
    """
    other_flip_turns = [t for t in range(1, T)
                        if t < len(seq_o) and seq_o[t] in LABEL_SET
                        and seq_o[t - 1] in LABEL_SET and seq_o[t] != seq_o[t - 1]]
    isolation_ref = _own_label_change_turns(seq_s, T) if seq_s is not None else flip_ts_raw
    cleaned = []
    n_coflip_dropped = 0
    for t in flip_ts_raw:
        if exclude_coflips and any(abs(t - o) <= clean_window for o in other_flip_turns):
            n_coflip_dropped += 1
            continue
        if any(t2 != t and abs(t - t2) <= clean_window for t2 in isolation_ref):
            continue
        cleaned.append(t)
    return cleaned, n_coflip_dropped


def _fit_turn_balanced_direction(train_conv_list, direction_feature,
                                 rng_bal, balanced_max_turn, min_samples,
                                 agree_conv_list=None,
                                 balance=True, reweight_across_turns=True,
                                 classifier="ridge_lda"):
    """
    Shared turn-balanced agree/disagree direction fit; "agree" comes from the
    SEPARATE, required agree_conv_list. Returns {"u", "mu", "lam", ...} or None.
    """
    if direction_feature == "agree_disagree":
        by_turn_a, by_turn_b = defaultdict(list), defaultdict(list)
        limit_of = lambda T: T if balanced_max_turn is None else min(T, balanced_max_turn + 1)

        def _harvest(convs, target_is_pos, sink):
            """Collects turns 1..k-1 into `sink`, where k is the first turn the speaker's
            own label changes or is_pos=(ls[t]==lo[t-1]) stops matching `target_is_pos`."""
            for conv in convs:
                ls, lo, hs, T = conv['seq_s'], conv['seq_o'], conv['hs'], conv['T']
                limit = limit_of(T)
                for t in range(1, limit):
                    if t not in hs:
                        continue
                    if ls[t] not in LABEL_SET:
                        break
                    if ls[t - 1] not in LABEL_SET or lo[t - 1] not in LABEL_SET:
                        break
                    if ls[t] != ls[t - 1]:
                        break   # speaker's own opinion changed -- stop for good
                    if (ls[t] == lo[t - 1]) != target_is_pos:
                        break   # first transition away from the target relation -- stop for good
                    sink[t].append(hs[t])

        # "disagree" class: disagree-starting TRAIN population (is_pos False).
        _harvest(train_conv_list, False, by_turn_b)
        # "agree" class: the SEPARATE agree_conv_list (is_pos True) -- the
        # mirror of why disagree-starting's turn 1 is always "disagree".
        _harvest(agree_conv_list or [], True, by_turn_a)
        class_names = ("agree", "disagree")
    else:
        raise ValueError(f"unknown direction_feature: {direction_feature!r}")

    # Raw per-class totals, before balancing.
    n_a_raw = sum(len(v) for v in by_turn_a.values())
    n_b_raw = sum(len(v) for v in by_turn_b.values())
    per_turn_raw = {t: (len(by_turn_a.get(t, [])), len(by_turn_b.get(t, [])))
                    for t in sorted(set(by_turn_a) | set(by_turn_b))}

    fit_a, fit_b, turn_a, turn_b = [], [], [], []
    if balance:
        per_turn_n = {}
        for t in sorted(set(by_turn_a) | set(by_turn_b)):
            a_t, b_t = by_turn_a.get(t, []), by_turn_b.get(t, [])
            n_t = min(len(a_t), len(b_t))
            if n_t < 1:
                continue
            per_turn_n[t] = n_t
            idx_a = rng_bal.choice(len(a_t), size=n_t, replace=False)
            idx_b = rng_bal.choice(len(b_t), size=n_t, replace=False)
            fit_a.extend(a_t[i] for i in idx_a)
            fit_b.extend(b_t[i] for i in idx_b)
            turn_a.extend([t] * n_t)
            turn_b.extend([t] * n_t)
    else:
        # UNBALANCED -- every raw harvested example, no per-turn downsampling.
        per_turn_n = None
        for t in sorted(set(by_turn_a) | set(by_turn_b)):
            a_t, b_t = by_turn_a.get(t, []), by_turn_b.get(t, [])
            fit_a.extend(a_t); turn_a.extend([t] * len(a_t))
            fit_b.extend(b_t); turn_b.extend([t] * len(b_t))

    n_fit_a, n_fit_b = len(fit_a), len(fit_b)
    if n_fit_a < 1 or n_fit_b < 1 or n_fit_a + n_fit_b < min_samples:
        return None
    h_train = np.stack(fit_a + fit_b)
    f_train = np.array([1.0] * n_fit_a + [-1.0] * n_fit_b)
    turn_train = np.array(turn_a + turn_b, dtype=float)
    mu = h_train.mean(0)
    # reweight_across_turns: every distinct turn contributes equal total
    # weight, on top of whichever pool was assembled above.
    sample_weight = _turn_reweight(turn_train) if reweight_across_turns else None
    if classifier == "ridge_lda":
        _dir_fit = _ridge_lda_dir
    elif classifier == "logreg":
        _dir_fit = _logreg_dir
    else:
        raise ValueError(f"unknown classifier: {classifier!r}")
    res = _dir_fit(h_train - mu, f_train, sample_weight=sample_weight)
    if res is None:
        return None
    u, lam = res
    return {"u": u, "mu": mu, "lam": lam,
            "n_train_total": n_fit_a, "n_fit_a": n_fit_a, "n_fit_b": n_fit_b,
            "per_turn_n": per_turn_n, "class_names": class_names,
            "n_a_raw": n_a_raw, "n_b_raw": n_b_raw, "per_turn_raw": per_turn_raw}


def _markov_p_agree(nturn, pi_dis, lam2):
    """P(agree at absolute turn `nturn`) for a disagree-starting two-state Markov
    chain: closed-form 1 - [pi_dis + (1-pi_dis)*lam2**nturn]."""
    p_dis = pi_dis + (1.0 - pi_dis) * (lam2 ** nturn)
    return 1.0 - p_dis


def _fit_dataset_markov(results_dir, fpaths_ds, allowed_scen_keys):
    """Closed-form Markov fit (a, b, lam2, pi_dis) scoped to one dataset's disagree-
    starting conversations, restricted to `allowed_scen_keys` (the caller's own TRAIN split)."""
    state_seqs = []
    for fpath in fpaths_ds:
        summary = _load_summary(fpath)
        for scen_key, data in summary.items():
            if scen_key not in allowed_scen_keys:
                continue
            labels_p1 = data.get("labels_p1", [])
            labels_p2 = data.get("labels_p2", [])
            T_raw = min(len(labels_p1), len(labels_p2))
            # Longest turn-aligned prefix where both speakers have a parseable label.
            T = 0
            while T < T_raw and labels_p1[T] in LABEL_SET and labels_p2[T] in LABEL_SET:
                T += 1
            if T < 3:
                continue
            # 1 = agree, 0 = disagree -- see markov_validation's docstring.
            state = np.array([labels_p1[t] == labels_p2[t] for t in range(T)], dtype=float)
            if state[0] != 0:   # keep only disagree-starting (state[0] == disagree == 0)
                continue
            state_seqs.append(state)
    if len(state_seqs) < 6:
        return None
    counts = np.zeros((2, 2))
    for seq in state_seqs:
        for t in range(len(seq) - 1):
            counts[int(seq[t]), int(seq[t + 1])] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    T_mat = counts / row_sums
    # a=P(agree->disagree), b=P(disagree->agree) -- with 1=agree/0=disagree,
    # that's T_mat[1,0]/T_mat[0,1].
    a_mk = float(T_mat[1, 0]); b_mk = float(T_mat[0, 1])
    lam2 = 1.0 - a_mk - b_mk
    pi_dis = a_mk / (a_mk + b_mk) if (a_mk + b_mk) > 1e-12 else 0.5
    return {"a": a_mk, "b": b_mk, "lam2": lam2, "pi_dis": pi_dis, "n_conv": len(state_seqs)}


def fit_markov_powerlaw_vs_turn_baseline(
        results_dir, file_suffix,
        layer_idx=2, inner_key="inners",
        skip_models=(),
        direction_feature="agree_disagree",
        min_samples=5,
        seed=42, train_frac=0.7,
        balance=True,
        balanced_max_turn=None,
        reweight_across_turns=True,
        classifier="ridge_lda",
        out_filename="markov_powerlaw_fit_agree_disagree_last_by_p.pdf"):
    """
    Fits z ~= alpha * p_agree(n)**gamma + beta per (model, dataset) and plots
    it against held-out TEST per-turn-number z means, one panel per model,
    x-axis = p(t) (the Markov-predicted agreement probability), not raw
    turn number.
    """
    model_files = _load_model_files(results_dir, file_suffix, skip_models)
    if not model_files:
        return {}

    all_results = {}

    for model_name, mfiles in sorted(model_files.items()):
        fpaths = sorted(mfiles)
        if not fpaths:
            continue

        conv_list = _collect_conversations(
            fpaths, results_dir, inner_key, layer_idx, tag_ds=True,
            scenario_filter=_disagree_starting)
        n_conv = len(conv_list)
        if n_conv < 2:
            continue

        # Used only for the train/test row-count sanity gate below -- the
        # actual fit population is sourced separately below (cross-population).
        all_conv_i = []
        for conv_idx, conv in enumerate(conv_list):
            hs, seq_s, T = conv['hs'], conv['seq_s'], conv['T']
            limit = T if balanced_max_turn is None else min(T, balanced_max_turn + 1)
            for t in range(1, limit):
                if t not in hs:
                    continue
                ls_prev = seq_s[t - 1]; ls_curr = seq_s[t]
                if ls_prev not in LABEL_SET or ls_curr not in LABEL_SET:
                    continue
                all_conv_i.append(conv_idx)

        n = len(all_conv_i)
        if n < 20:
            continue

        conv_i_arr = np.array(all_conv_i)

        train_idx, test_idx, train_conv_list, test_conv_list = _train_test_split_convs(
            conv_list, seed=seed, train_frac=train_frac)
        train_mask = np.array([c in train_idx for c in conv_i_arr])
        n_train, n_test = int(train_mask.sum()), int((~train_mask).sum())
        if n_train < 20 or n_test < 20:
            continue

        if direction_feature == "agree_disagree":
            # Cross-population sourcing: "agree" from a separate agree-starting
            # population (TRAIN half), "disagree" from the disagree-starting TRAIN.
            agree_conv_list = _collect_conversations(
                fpaths, results_dir, inner_key, layer_idx, tag_ds=True,
                scenario_filter=_agree_starting)
            _, _, agree_conv_list, _ = _train_test_split_convs(
                agree_conv_list, seed=seed, train_frac=train_frac)
            dir_h, dir_feat, dir_turn = [], [], []
            def _harvest_powerlaw(convs, target_is_pos, feat_val):
                for conv in convs:
                    ls, lo, hs, T = conv['seq_s'], conv['seq_o'], conv['hs'], conv['T']
                    limit = T if balanced_max_turn is None else min(T, balanced_max_turn + 1)
                    for t in range(1, limit):
                        if t not in hs:
                            continue
                        if ls[t] not in LABEL_SET:
                            break
                        if ls[t - 1] not in LABEL_SET or lo[t - 1] not in LABEL_SET:
                            break
                        if ls[t] != ls[t - 1]:
                            break   # speaker's own opinion changed -- stop for good
                        if (ls[t] == lo[t - 1]) != target_is_pos:
                            break
                        dir_h.append(hs[t]); dir_feat.append(feat_val); dir_turn.append(float(t))
            _harvest_powerlaw(agree_conv_list, True, 1.0)
            _harvest_powerlaw(train_conv_list, False, -1.0)
            if len(dir_h) < 20:
                continue
            R_dir    = np.stack(dir_h)
            feat_dir = np.array(dir_feat)
            turn_dir = np.array(dir_turn)
        else:
            raise ValueError(f"unknown direction_feature: {direction_feature!r}")

        if balance:
            bal_idx = _balance_by_turn(feat_dir, turn_dir, seed=seed)
            if len(bal_idx) < 20:
                continue
            mu = R_dir[bal_idx].mean(0)
            R_c_fit = R_dir[bal_idx] - mu
            fit_feat = feat_dir[bal_idx]
            fit_turn = turn_dir[bal_idx]
        else:
            # UNBALANCED -- see docstring: no _balance_by_turn, unless
            # balance=True was explicitly passed.
            mu = R_dir.mean(0)
            R_c_fit = R_dir - mu
            fit_feat = feat_dir
            fit_turn = turn_dir

        sample_weight = _turn_reweight(fit_turn) if reweight_across_turns else None
        if classifier == "ridge_lda":
            _dir_fit = _ridge_lda_dir
        elif classifier == "logreg":
            _dir_fit = _logreg_dir
        else:
            raise ValueError(f"unknown classifier: {classifier!r}")
        res = _dir_fit(R_c_fit, fit_feat, sample_weight=sample_weight)
        if res is None:
            continue
        u, lam = res

        def _collect_turns(convs, ds_name):
            t_conv_i, t_n, t_z = [], [], []
            for ci, conv in enumerate(convs):
                if conv['ds'] != ds_name:
                    continue
                z_by_t = {t: float((h - mu) @ u) for t, h in conv['hs'].items()}
                for t, z in z_by_t.items():
                    t_conv_i.append(ci); t_n.append(t); t_z.append(z)
            return (np.array(t_conv_i), np.array(t_n, dtype=float), np.array(t_z, dtype=float))

        ds_names = sorted({conv['ds'] for conv in conv_list})
        model_ds_results = {}
        for ds_name in ds_names:
            fpaths_ds = [f for f in fpaths if _dataset_name(f) == ds_name]
            train_scen_keys = {c['scen_key'] for c in train_conv_list if c['ds'] == ds_name}
            test_scen_keys  = {c['scen_key'] for c in test_conv_list  if c['ds'] == ds_name}
            if not train_scen_keys or not test_scen_keys:
                continue

            markov = _fit_dataset_markov(results_dir, fpaths_ds, train_scen_keys)
            if markov is None:
                continue

            train_conv_i, train_n, train_z = _collect_turns(train_conv_list, ds_name)
            test_conv_i,  test_n,  test_z  = _collect_turns(test_conv_list, ds_name)

            # exclude turn 0 -- answered "cold," before either speaker responds.
            train_keep = train_n > 0
            train_n, train_z = train_n[train_keep], train_z[train_keep]
            test_keep = test_n > 0
            test_n, test_z = test_n[test_keep], test_z[test_keep]
            if len(train_z) == 0 or len(test_z) == 0:
                continue

            train_p = np.array([_markov_p_agree(t, markov["pi_dis"], markov["lam2"])
                               for t in train_n])
            fit_markov = _fit_powerlaw(train_p, train_z, min_samples=min_samples, seed=seed)
            if fit_markov is None:
                continue

            # per-turn-number means on TEST data only -- the dots the
            # fitted curve is plotted against.
            turn_ns, turn_mean_z, turn_se_z = [], [], []
            for tn in sorted(set(test_n.astype(int))):
                in_t = test_n == tn
                turn_ns.append(tn)
                turn_mean_z.append(float(test_z[in_t].mean()))
                turn_se_z.append(float(test_z[in_t].std() / np.sqrt(in_t.sum())))
            turn_ns_arr, turn_mean_z_arr = np.array(turn_ns), np.array(turn_mean_z)

            n_min = float(min(train_n.min(), test_n.min()))
            n_max = float(max(train_n.max(), test_n.max()))
            n_curve = np.linspace(n_min, n_max, 200)
            p_curve = np.array([_markov_p_agree(x, markov["pi_dis"], markov["lam2"])
                               for x in n_curve])
            z_curve_markov = _naive_powerlaw(p_curve, fit_markov["alpha"],
                                             fit_markov["beta"], fit_markov["gamma"])
            # p(t) at each plotted dot's turn number -- x-axis for the
            # by-p(t) plot below, same turns the dots are already scored at.
            p_at_dots = np.array([_markov_p_agree(tn, markov["pi_dis"], markov["lam2"])
                                  for tn in turn_ns_arr])

            model_ds_results[ds_name] = {
                "markov": markov, "fit_markov": fit_markov,
                "p_curve": p_curve, "z_curve_markov": z_curve_markov,
                "turn_mean_z": turn_mean_z_arr,
                "turn_se_z": np.array(turn_se_z), "p_at_dots": p_at_dots,
            }

        if model_ds_results:
            all_results[model_name] = model_ds_results

    if not all_results:
        return all_results

    # ── plot: one panel per model, one color per dataset -- fitted Markov
    # p(t) curve vs. held-out TEST per-turn-number z means ─────────────────
    models = sorted(all_results.keys())
    n_models = len(models)
    palette = ["#0072B2", "#CC79A7", "#009E73", "#E69F00", "#56B4E9", "#44AA99"]
    all_ds_names = sorted({ds for r in all_results.values() for ds in r})
    ds_color = {ds: palette[i % len(palette)] for i, ds in enumerate(all_ds_names)}

    fig, axes = plt.subplots(1, n_models, figsize=(5.2 * n_models, 4.6), squeeze=False)
    for mi, mname in enumerate(models):
        ax = axes[0][mi]
        ds_results = all_results[mname]
        for ds_name in sorted(ds_results):
            v = ds_results[ds_name]
            color = ds_color[ds_name]
            ax.errorbar(v["p_at_dots"], v["turn_mean_z"], yerr=v["turn_se_z"], fmt="o",
                       color=color, ms=5, capsize=3, alpha=0.85)
            ax.plot(v["p_curve"], v["z_curve_markov"], "-", color=color, lw=1.8, alpha=0.8)
        if mi == 0:
            ax.set_ylabel(r"Aggregated Internal Agreement ($\bar{Z}$)", fontsize=18, rotation=90, ha="center", va="center", labelpad=14, y=0.4)
        ax.tick_params(axis='both', labelsize=16)
        ax.set_title(_short_model_name(mname), fontsize=20)

    # One shared x-axis label below the whole row instead of repeating
    # "Behavioral Markov Agreement (p(t))" under every panel.
    fig.text(0.5, -0.02, "Behavioral Markov Agreement (p(t))", ha="center", va="top", fontsize=18)

    # Dataset entries use a circular marker, matching the actual errorbar
    # dots' "o" shape in the plot itself.
    legend_handles = (
        [plt.Line2D([0], [0], marker="o", linestyle="None", color=ds_color[ds],
                    markersize=10, label=_short_dataset_name(ds)) for ds in all_ds_names]
        + [plt.Line2D([0], [0], color="gray", lw=1.8, linestyle="-", label="Markov p(t) fit")]
    )
    _ncol = int(np.ceil(len(legend_handles) / 2))
    fig.legend(handles=legend_handles, loc="lower center", ncol=_ncol,
              bbox_to_anchor=(0.5, -0.30), fontsize=17, frameon=True)
    plt.tight_layout()
    out = os.path.join(results_dir, out_filename)
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)

    return all_results


def analyze_z_absolute_trajectories(
        results_dir, file_suffix,
        layer_idx=2, inner_key="inners",
        skip_models=(),
        min_samples=5,
        seed=42, train_frac=0.7,
        balance=True,
        reweight_across_turns=True,
        n_examples=10,
        min_turn_population=5,
        classifier="ridge_lda"):
    """
    Collects what plot_z_absolute_trajectories_individual needs: per (model, dataset),
    the population mean z(t) curve and `n_examples` random individual z(t) trajectories.
    """
    model_files = _load_model_files(results_dir, file_suffix, skip_models)
    if not model_files:
        return {}

    all_results = {}
    rng = np.random.default_rng(seed)

    for model_name, mfiles in sorted(model_files.items()):
        fpaths = sorted(mfiles)
        if not fpaths:
            continue

        # _collect_conversations_with_text: identical filtering/ordering as
        # _collect_conversations, plus 'file_key'/'speaker_idx' for real message text.
        conv_list, text_by_key = _collect_conversations_with_text(
            fpaths, results_dir, inner_key, layer_idx,
            scenario_filter=_disagree_starting)
        n_conv = len(conv_list)
        if n_conv < 2:
            continue

        train_idx, test_idx, train_conv_list, test_conv_list = _train_test_split_convs(
            conv_list, seed=seed, train_frac=train_frac)
        if len(train_conv_list) < 2 or len(test_conv_list) < 2:
            continue

        # agree_disagree direction fit -- same harvest/balance/reweight
        # machinery as fit_markov_powerlaw_vs_turn_baseline's agree_disagree branch.
        agree_conv_list = _collect_conversations(
            fpaths, results_dir, inner_key, layer_idx, tag_ds=True,
            scenario_filter=_agree_starting)
        _, _, agree_conv_list, _ = _train_test_split_convs(
            agree_conv_list, seed=seed, train_frac=train_frac)

        dir_h, dir_feat, dir_turn = [], [], []

        def _harvest(convs, target_is_pos, feat_val):
            for conv in convs:
                ls, lo, hs, T = conv['seq_s'], conv['seq_o'], conv['hs'], conv['T']
                for t in range(1, T):
                    if t not in hs:
                        continue
                    if ls[t] not in LABEL_SET:
                        break
                    if ls[t - 1] not in LABEL_SET or lo[t - 1] not in LABEL_SET:
                        break
                    if ls[t] != ls[t - 1]:
                        break
                    if (ls[t] == lo[t - 1]) != target_is_pos:
                        break
                    dir_h.append(hs[t]); dir_feat.append(feat_val); dir_turn.append(float(t))

        _harvest(agree_conv_list, True, 1.0)
        _harvest(train_conv_list, False, -1.0)
        if len(dir_h) < 20:
            continue
        R_dir    = np.stack(dir_h)
        feat_dir = np.array(dir_feat)
        turn_dir = np.array(dir_turn)

        if balance:
            bal_idx = _balance_by_turn(feat_dir, turn_dir, seed=seed)
            if len(bal_idx) < 20:
                continue
            mu = R_dir[bal_idx].mean(0)
            R_c_fit = R_dir[bal_idx] - mu
            fit_feat = feat_dir[bal_idx]
            fit_turn = turn_dir[bal_idx]
        else:
            mu = R_dir.mean(0)
            R_c_fit = R_dir - mu
            fit_feat = feat_dir
            fit_turn = turn_dir

        sample_weight = _turn_reweight(fit_turn) if reweight_across_turns else None
        if classifier == "ridge_lda":
            _dir_fit = _ridge_lda_dir
        elif classifier == "logreg":
            _dir_fit = _logreg_dir
        else:
            raise ValueError(f"unknown classifier: {classifier!r}")
        res = _dir_fit(R_c_fit, fit_feat, sample_weight=sample_weight)
        if res is None:
            continue
        u, lam = res

        # Per DATASET: population mean z(t) (that dataset's TEST convs only) +
        # n_examples random individuals; direction (u, mu) stays pooled across datasets.
        ds_names = sorted({conv['ds'] for conv in conv_list})
        model_result = {}
        for ds_name in ds_names:
            test_ds_convs = [c for c in test_conv_list if c['ds'] == ds_name]
            if len(test_ds_convs) < min_samples:
                continue

            z_by_turn_pop = defaultdict(list)
            conv_zs = []
            for conv in test_ds_convs:
                # turn 0 excluded -- answered "cold," before either speaker responds.
                z_by_t = {t: float((h - mu) @ u) for t, h in conv['hs'].items() if t >= 1}
                if not z_by_t:
                    continue   # only turn 0 was decodable -- nothing left to show
                for t, z in z_by_t.items():
                    z_by_turn_pop[t].append(z)
                # kept alongside its conv so a sampled example can recover its real text.
                conv_zs.append((z_by_t, conv))

            mean_turns = sorted(t for t, zs in z_by_turn_pop.items()
                                if len(zs) >= min_turn_population)
            if len(mean_turns) < 2:
                continue
            mean_z = [float(np.mean(z_by_turn_pop[t])) for t in mean_turns]

            # Shared per-conversation entry builder, used for BOTH `examples`
            # (random sample) and `all_trajectories` (every TEST conversation).
            def _build_entry(z_by_t, conv):
                turns = sorted(z_by_t)
                ls, lo = conv['seq_s'], conv['seq_o']
                speaker_idx = conv['speaker_idx']
                text_entry = text_by_key.get(conv['file_key'], {})
                conversation_text = text_entry.get("conversation", [])
                self_text, other_text, self_label, other_label = [], [], [], []
                for t in turns:
                    self_i, other_i = 2 * t + speaker_idx, 2 * t + (1 - speaker_idx)
                    self_text.append(conversation_text[self_i]
                                     if self_i < len(conversation_text) else None)
                    other_text.append(conversation_text[other_i]
                                      if other_i < len(conversation_text) else None)
                    self_label.append(ls[t] if t < len(ls) else None)
                    other_label.append(lo[t] if t < len(lo) else None)
                return {"turns": turns, "z": [z_by_t[t] for t in turns],
                       "self_text": self_text, "other_text": other_text,
                       "self_label": self_label, "other_label": other_label,
                       "scen_key": conv['scen_key']}

            n_pick = min(n_examples, len(conv_zs))
            pick_idx = rng.choice(len(conv_zs), size=n_pick, replace=False)
            examples = [_build_entry(*conv_zs[int(i)]) for i in pick_idx]

            # all_trajectories: same shape as `examples`, but EVERY TEST
            # conversation of this dataset, for callers needing the full population.
            all_trajectories = [_build_entry(z_by_t, conv) for z_by_t, conv in conv_zs]
            model_result[ds_name] = {"mean_turns": mean_turns, "mean_z": mean_z,
                                     "examples": examples,
                                     "z_by_turn_pop": dict(z_by_turn_pop),
                                     "all_trajectories": all_trajectories}

        if model_result:
            all_results[model_name] = model_result

    return all_results


def plot_z_absolute_trajectories_individual(
        all_results, results_dir,
        out_filename="z_absolute_trajectories_individual.pdf"):
    """
    Grid of panels (rows=model, cols=dataset) of individual real TEST conversations'
    z(t) trajectories (each a different color) vs. absolute turn number.
    """
    model_names = sorted(all_results.keys())
    if not model_names:
        return
    ds_names = sorted({ds for m in all_results.values() for ds in m.keys()})

    fig, axes = plt.subplots(len(model_names), len(ds_names),
                             figsize=(4.2 * len(ds_names), 3.6 * len(model_names)),
                             squeeze=False)
    cmap = plt.get_cmap("tab10")

    # Label once per row/column, not per panel: dataset title above the
    # top-most row with data, short model name left of the left-most column.
    n_rows, n_cols = len(model_names), len(ds_names)
    has_data = [[all_results[model_names[ri]].get(ds_names[ci]) is not None
                for ci in range(n_cols)] for ri in range(n_rows)]
    top_row_for_col = [min((ri for ri in range(n_rows) if has_data[ri][ci]), default=None)
                       for ci in range(n_cols)]
    left_col_for_row = [min((ci for ci in range(n_cols) if has_data[ri][ci]), default=None)
                        for ri in range(n_rows)]
    # Global (not per-panel) max turn, so every panel gets the same even-numbered ticks.
    global_max_T = max((t for m in all_results.values() for r in m.values() if r is not None
                        for ex in r["examples"] for t in ex["turns"]), default=0)
    xticks = np.arange(2, global_max_T + 1, 2)

    for i, model_name in enumerate(model_names):
        for j, ds_name in enumerate(ds_names):
            ax = axes[i][j]
            r = all_results[model_name].get(ds_name)
            if r is None:
                ax.axis("off"); continue
            for k, ex in enumerate(r["examples"]):
                ax.plot(ex["turns"], ex["z"], color=cmap(k % 10), lw=2.2, alpha=0.9,
                       marker="o", markersize=3)
            ax.set_yticks([])
            ax.set_xticks(xticks)
            ax.tick_params(axis='x', labelsize=20)
            if i == top_row_for_col[j]:
                ax.set_title(_short_dataset_name(ds_name), fontsize=22)
            if j == left_col_for_row[i]:
                ax.set_ylabel(_short_model_name(model_name), fontsize=22,
                             rotation=90, ha="center", va="center", labelpad=10)
    # The per-panel title/ylabel slots hold the dataset/model labels, not the
    # actual axis units -- state those once, outside the grid.
    fig.text(0.5, -0.02, "Turn Number", ha="center", va="top", fontsize=24)
    fig.text(-0.02, 0.5, "Internal Agreement (z)", ha="right", va="center",
            rotation=90, fontsize=24)
    fig.tight_layout()
    out_path = os.path.join(results_dir, out_filename)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the analysis over results/ and write the paper's figures.")
    parser.add_argument("--results_dir", default="results",
                       help="Directory holding generate_data_submission.py's JSON output "
                            "(default: results)")
    parser.add_argument("--n_examples", type=int, nargs="+", default=[300, 206],
                       help="Scenario count(s) in the filenames to analyze. A dataset is "
                            "saved under however many scenarios it actually yielded, so the "
                            "paper's runs are 300 for every dataset except Global, which "
                            "caps at 206 -- hence two values by default. Pass a single "
                            "value (e.g. --n_examples 50) if every dataset has the same "
                            "count.")
    parser.add_argument("--conv_length", type=int, default=10,
                       help="Turns per conversation, as passed to generate_data_submission.py "
                            "(default: 10)")
    parser.add_argument("--layer_idx", type=int, default=2,
                       help="Which of the 4 saved layers to read (default: 2, the 85%% depth "
                            "layer used in the paper)")
    args = parser.parse_args()

    RESULTS_DIR = args.results_dir
    SUFFIXES    = [f"{n}ex_{args.conv_length}_discussion_False_False_False"
                   for n in args.n_examples]
    LAYER_IDX   = args.layer_idx
    SKIP_MODELS = ()

    # Fail loudly here rather than letting every analysis below silently
    # return {} and write nothing.
    if not _load_model_files(RESULTS_DIR, SUFFIXES, SKIP_MODELS):
        raise SystemExit(
            f"No result files found in {RESULTS_DIR!r} matching any of: "
            + ", ".join(f"*_{s}.json" for s in SUFFIXES)
            + "\nCheck --results_dir, and that --n_examples/--conv_length match the "
              "scenario counts actually in the filenames.")

    z_traj_results = analyze_z_absolute_trajectories(
        results_dir=RESULTS_DIR, file_suffix=SUFFIXES,
        layer_idx=LAYER_IDX, inner_key="inners_last",
        skip_models=SKIP_MODELS)
    plot_z_absolute_trajectories_individual(
        z_traj_results, results_dir=RESULTS_DIR,
        out_filename="z_absolute_trajectories_individual.pdf")

    markov_results_disagree_starting = markov_validation(
        results_dir=RESULTS_DIR, file_suffix=SUFFIXES,
        skip_models=SKIP_MODELS)
    plot_markov_2nd_order_baseline(
        markov_results_disagree_starting, results_dir=RESULTS_DIR,
        out_filename="markov_validation_2nd_order_agree_starting.pdf",
        y_label="Agreement (%)")

    fit_markov_powerlaw_vs_turn_baseline(
        results_dir=RESULTS_DIR, file_suffix=SUFFIXES,
        skip_models=SKIP_MODELS,
        layer_idx=LAYER_IDX, inner_key="inners_last",
        out_filename="markov_powerlaw_fit_agree_disagree_last_by_p.pdf")

    nocoflip_results = estimate_pooled_flip_signal_turn0dir(
        results_dir=RESULTS_DIR, file_suffix=SUFFIXES,
        layer_idx=LAYER_IDX, inner_key="inners_last",
        skip_models=SKIP_MODELS,
        exclude_coflips=True, clean_window=2, require_single_flip=True)
    plot_chance_distance_postflip_heatmap(
        nocoflip_results, results_dir=RESULTS_DIR, lags=(-1, 0, 1, 2, 3),
        out_filename="pooled_flip_signal_turn0dir_last_nocoflip_heatmap_postflip.pdf")

