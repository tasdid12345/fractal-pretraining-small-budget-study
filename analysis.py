"""Fractal pre-training stand-in on sklearn digits.

Baseline: MLP pre-trained on a random budget of IFS fractal images, then fine-tuned.
Improved: same budget, but drawn from a larger candidate pool by k-means diversity.
"""
import json, os, sys, time, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import f1_score, accuracy_score

SEED = 42
SEEDS = [42, 43, 44]
N_CLASSES = 10        # IFS classes = digit classes, so the output head can be warm-started
N_BUDGET = 2000       # fractal images used for pre-training (same for both models)
POOL_FACTOR = 5       # improved model selects its budget from a pool this many times larger
SHOTS = 20            # labelled digits per class used for fine-tuning
K_MAPS = 3            # affine maps per IFS
N_POINTS = 400        # chaos-game points per fractal image
L = 5.0               # half-width of the rendering window (contractive maps keep points inside)
URL = "https://scikit-learn.org/stable/datasets/toy_dataset.html#optical-recognition-of-handwritten-digits-dataset"


def make_ifs_classes(rng):
    """Fixed per-class IFS definitions: contractive 2x2 matrices (spectral norm <= 0.8) and translations."""
    A = rng.uniform(-1, 1, (N_CLASSES, K_MAPS, 2, 2))
    norms = np.linalg.norm(A, ord=2, axis=(2, 3), keepdims=True)
    return A * np.minimum(1, 0.8 / norms), rng.uniform(-1, 1, (N_CLASSES, K_MAPS, 2))


def render_fractals(labels, A, T, rng):
    """Chaos-game rendering of a batch of IFS fractals (vectorised over the batch) -> (B, 64) in [0, 1]."""
    B = len(labels)
    Ai = A[labels] + rng.normal(0, 0.03, (B, K_MAPS, 2, 2))  # per-sample jitter around the class IFS
    Ti = T[labels] + rng.normal(0, 0.03, (B, K_MAPS, 2))
    x = rng.uniform(-1, 1, (B, 2))
    img, rows = np.zeros((B, 64)), np.arange(B)
    for step in range(N_POINTS + 20):
        k = rng.integers(K_MAPS, size=B)
        x = np.einsum("bij,bj->bi", Ai[rows, k], x) + Ti[rows, k]
        if step >= 20:  # discard burn-in points, then bin onto the 8x8 grid
            r = np.clip(np.floor((x[:, 1] + L) / (2 * L) * 8).astype(int), 0, 7)
            c = np.clip(np.floor((x[:, 0] + L) / (2 * L) * 8).astype(int), 0, 7)
            np.add.at(img, (rows, r * 8 + c), 1)
    img = np.log1p(img)
    return img / np.maximum(img.max(axis=1, keepdims=True), 1e-9)


def select_diverse(pool_X, n, rng):
    """k-means diversity: one pool image per cluster (the member nearest its centroid); fill gaps randomly."""
    km = MiniBatchKMeans(n_clusters=n, random_state=SEED, n_init=1, batch_size=1024).fit(pool_X)
    d = km.transform(pool_X)  # distances from every pool image to every centroid
    idx = []
    for j in range(n):
        members = np.where(km.labels_ == j)[0]
        if len(members):
            idx.append(members[np.argmin(d[members, j])])
    idx = np.array(idx, dtype=int)
    if len(idx) < n:  # rare empty clusters: top up with random unused pool images
        rest = np.setdiff1d(np.arange(len(pool_X)), idx)
        idx = np.concatenate([idx, rng.choice(rest, n - len(idx), replace=False)])
    return idx


def run(method, seed, ifs, data):
    """Pre-train (timed) on fractals, then fine-tune on labelled digits and evaluate on the test split."""
    Xft, yft, Xte, yte = data
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    if method == "baseline":  # random budget of fractal images
        y_pre = rng.integers(N_CLASSES, size=N_BUDGET)
        X_pre = render_fractals(y_pre, *ifs, rng)
    else:  # same budget, chosen by diversity from a larger pool
        pool_y = rng.integers(N_CLASSES, size=POOL_FACTOR * N_BUDGET)
        pool_X = render_fractals(pool_y, *ifs, rng)
        idx = select_diverse(pool_X, N_BUDGET, rng)
        X_pre, y_pre = pool_X[idx], pool_y[idx]
    # identical architecture and optimiser for both methods
    clf = MLPClassifier(hidden_layer_sizes=(128,), max_iter=50, random_state=seed, warm_start=True)
    clf.fit(X_pre, y_pre)
    pre_time = time.perf_counter() - t0  # includes generation and selection cost
    clf.set_params(max_iter=200)
    clf.fit(Xft, yft)  # warm-started fine-tuning on the labelled digits
    pred = clf.predict(Xte)
    return dict(f1=f1_score(yte, pred, average="macro"), acc=accuracy_score(yte, pred), time=pre_time)


def main():
    warnings.filterwarnings("ignore")
    digits = load_digits()
    X, y = digits.data / 16.0, digits.target
    # single stratified 70/30 split with fixed seed; test set used only at the end
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)
    rng = np.random.default_rng(SEED)
    ft_idx = np.concatenate([rng.choice(np.where(ytr == c)[0], SHOTS, replace=False) for c in range(10)])
    data = (Xtr[ft_idx], ytr[ft_idx], Xte, yte)
    ifs = make_ifs_classes(np.random.default_rng(SEED))

    res = {"baseline": [], "improved": []}
    for s in SEEDS:
        for m in res:
            print(f"[seed {s}] pre-training + fine-tuning {m} ...", flush=True)
            res[m].append(run(m, s, ifs, data))

    def agg(m, key):
        v = np.array([r[key] for r in res[m]], dtype=float)
        return float(v.mean()), float(v.std())

    b_f1, b_f1_sd = agg("baseline", "f1")
    i_f1, i_f1_sd = agg("improved", "f1")
    if not np.all(np.isfinite([b_f1, i_f1])) or b_f1 <= 0 or i_f1 <= 0:
        print("Could not produce meaningful metrics.", file=sys.stderr)
        sys.exit(1)

    results = {
        "question": "Can fractal pre-training with a small, cheaply generated synthetic budget, chosen by diversity, "
                    "match or beat fractal pre-training with a larger random budget on a small downstream "
                    "digit-recognition task?",
        "method": "IFS fractal pre-training stand-in for an MLP on sklearn digits; 70/30 stratified split; "
                  "20 labelled digits per class for fine-tuning; baseline pre-trains on a random fractal budget, "
                  "improved pre-trains on the same budget chosen by k-means diversity from a 5x larger pool; "
                  "seeds 42, 43, 44.",
        "dataset": {
            "name": "scikit-learn optical recognition of handwritten digits (load_digits)",
            "source_url": URL,
            "n_rows": int(len(X)), "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
            "same_as_paper": False, "baseline_is_papers_method": False,
        },
        "metric_name": "macro-F1 on held-out test split",
        "higher_is_better": True,
        "baseline": "MLP (128 hidden units) pre-trained on a random budget of 2000 IFS fractal images labelled by "
                    "IFS class, then fine-tuned on 200 labelled digits.",
        "improved": "Same MLP, same 2000-image budget, but the images are chosen by k-means diversity from a pool "
                    "of 10000 generated fractal images before identical pre-training and fine-tuning.",
        "metrics": {
            "baseline_score": b_f1, "baseline_score_std": b_f1_sd,
            "improved_score": i_f1, "improved_score_std": i_f1_sd,
            "improvement": i_f1 - b_f1,
            "baseline_f1": b_f1, "improved_f1": i_f1,
            "baseline_accuracy": agg("baseline", "acc")[0], "improved_accuracy": agg("improved", "acc")[0],
            "baseline_pretrain_seconds": agg("baseline", "time")[0],
            "improved_pretrain_seconds": agg("improved", "time")[0],
        },
        "notes": "Stand-in for the paper's fractal pre-training (not its ViT/CNN); MLPClassifier replaces it. "
                 "Budget is equal for both methods, so the 'small budget' part of the question is not tested directly. "
                 "Improved pre-training time includes pool generation and k-means, so it is not a speed-up. "
                 "k-means on pixels ignores labels, so class balance of the selected set is not enforced. "
                 "Hyper-parameters were fixed a priori, not tuned. Only 3 seeds and one digit split; "
                 "the test set was used once at the end.",
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.bar(["baseline", "improved"], [b_f1, i_f1], yerr=[b_f1_sd, i_f1_sd], capsize=6, color=["gray", "tab:blue"])
    ax.set_ylabel("macro-F1 (test)")
    ax.set_title("Fractal pre-training: random vs diverse budget")
    fig.tight_layout()
    fig.savefig("figures/macro_f1.png", dpi=120)
    print(f"baseline macro-F1 {b_f1:.4f} +/- {b_f1_sd:.4f} | improved {i_f1:.4f} +/- {i_f1_sd:.4f}")
    print("Results written to results.json")


if __name__ == "__main__":
    main()
