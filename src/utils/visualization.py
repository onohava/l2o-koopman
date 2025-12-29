import numpy as np
import matplotlib.pyplot as plt

def plot_results(
    fit_data,
    koopman_ae,
    reducer="mean",     # "mean" or "median"
    band="sem",         # None | "sem" | "std" | "iqr" | (lo_pct, hi_pct)
    title="Quadratic functions",
    use_log_y=True,
):
    A = np.asarray(fit_data)
    if A.ndim == 2:
        center = A
        low = high = None
    elif A.ndim == 3:
        # (trials, steps, opts) -> aggregate across trials axis=0
        if reducer == "median":
            center = np.nanmedian(A, axis=0)
        else:
            center = np.nanmean(A, axis=0)

        if band is None:
            low = high = None
        elif band == "sem":
            std = np.nanstd(A, axis=0)
            n = np.sum(~np.isnan(A), axis=0).clip(min=1)
            sem = std / np.sqrt(n)
            low, high = center - sem, center + sem
        elif band == "std":
            std = np.nanstd(A, axis=0)
            low, high = center - std, center + std
        elif band == "iqr":
            low = np.nanpercentile(A, 25, axis=0)
            high = np.nanpercentile(A, 75, axis=0)
        elif isinstance(band, (tuple, list)) and len(band) == 2:
            p_lo, p_hi = band
            low = np.nanpercentile(A, p_lo, axis=0)
            high = np.nanpercentile(A, p_hi, axis=0)
        else:
            raise ValueError("band must be None, 'sem', 'std', 'iqr', or (lo, hi) percentiles.")
    else:
        raise ValueError(f"fit_data must have 2 or 3 dims, got {A.shape}")

    steps, n_opts = center.shape
    x = np.arange(steps)

    colors = ['r', 'b', 'g', 'k', 'y']
    if n_opts > len(colors):
        cmap = plt.get_cmap("tab10")
        colors = [cmap(i % 10) for i in range(n_opts)]

    # Baselines + final L2O name
    base_names = globals().get("OPT_NAMES", [f"opt{i}" for i in range(n_opts - 1)])
    label_last = 'L2O with Koopman autoencoder' if koopman_ae else 'L2O'
    optimizer_names = list(base_names) + [label_last]
    if len(optimizer_names) != n_opts:
        optimizer_names = [f"opt{i}" for i in range(n_opts - 1)] + [label_last]

    plt.figure(figsize=(10, 6))
    for i, (name, color) in enumerate(zip(optimizer_names, colors)):
        linestyle = '-' if i == n_opts - 1 else '--'
        lw = 2.2 if i == n_opts - 1 else 1.6
        plt.plot(x, center[:, i], color=color, linestyle=linestyle, linewidth=lw, label=name)
        if low is not None and high is not None:
            plt.fill_between(x, low[:, i], high[:, i], color=color, alpha=0.15, linewidth=0)

    if use_log_y:
        plt.yscale('log')
    plt.xlabel('steps')
    plt.ylabel('loss')
    plt.title(title)
    plt.grid(True, which='both', alpha=0.25, linewidth=0.6)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()

def compare_results(
    fit_data_plain,
    fit_data_koopman,
    reducer="mean",     # "mean" or "median"
    band="sem",         # None | "sem" | "std" | "iqr" | (lo_pct, hi_pct)
    title="Quadratic: L2O vs L2O+KAE",
    use_log_y=True,
    lstm_index=None,    # if None, use last column
):
    A = np.asarray(fit_data_plain)
    B = np.asarray(fit_data_koopman)

    def _center_band(X):
        if X.ndim == 2:
            center = np.nanmean(X, axis=0) if reducer == "mean" else np.nanmedian(X, axis=0)
            return center, None, None
        if X.ndim != 3:
            raise ValueError(f"Expected (trials, steps, opts) or (steps, opts); got {X.shape}")

        if reducer == "median":
            center = np.nanmedian(X, axis=0)
        else:
            center = np.nanmean(X, axis=0)

        if band is None:
            return center, None, None
        if band == "sem":
            std = np.nanstd(X, axis=0)
            n = np.sum(~np.isnan(X), axis=0).clip(min=1)
            sem = std / np.sqrt(n)
            return center, center - sem, center + sem
        if band == "std":
            std = np.nanstd(X, axis=0)
            return center, center - std, center + std
        if band == "iqr":
            q25 = np.nanpercentile(X, 25, axis=0)
            q75 = np.nanpercentile(X, 75, axis=0)
            return center, q25, q75
        if isinstance(band, (tuple, list)) and len(band) == 2:
            lo, hi = band
            qlo = np.nanpercentile(X, lo, axis=0)
            qhi = np.nanpercentile(X, hi, axis=0)
            return center, qlo, qhi
        raise ValueError("band must be None, 'sem', 'std', 'iqr', or (lo, hi).")

    cA, lA, hA = _center_band(A)
    cB, lB, hB = _center_band(B)

    if cA.shape != cB.shape:
        raise ValueError(f"Shape mismatch: plain {cA.shape} vs KAE {cB.shape}")

    steps, n_opts = cA.shape
    idx = n_opts - 1 if lstm_index is None else int(lstm_index)
    x = np.arange(steps)

    plt.figure(figsize=(10, 6))

    # L2O (plain)
    plt.plot(x, cA[:, idx], color='r', linestyle='-.', linewidth=2.2, label='L2O')
    if lA is not None and hA is not None:
        plt.fill_between(x, lA[:, idx], hA[:, idx], color='r', alpha=0.12, linewidth=0)

    # L2O + KAE
    plt.plot(x, cB[:, idx], color='b', linestyle='-', linewidth=2.6, label='L2O + KAE')
    if lB is not None and hB is not None:
        plt.fill_between(x, lB[:, idx], hB[:, idx], color='b', alpha=0.15, linewidth=0)

    if use_log_y:
        plt.yscale('log')
    plt.xlabel('steps')
    plt.ylabel('loss')
    plt.title(title)
    plt.grid(True, which='both', alpha=0.25, linewidth=0.6)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()
