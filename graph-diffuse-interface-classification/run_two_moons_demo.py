"""
Demo: classify the classic "Two Moons" dataset using the graph diffuse
interface model, with only a handful of labelled points per class.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

from diffuse_interface_classifier import classify


def main():
    X, y_true = make_moons(n_samples=300, noise=0.08, random_state=0)
    y_true = 2 * y_true - 1  # convert {0,1} -> {-1,+1}

    # Only reveal a handful of labels per class ("semi-supervised" setting)
    rng = np.random.default_rng(0)
    n_labeled_per_class = 3
    labels_known = {}
    for cls in (-1, 1):
        idx = np.where(y_true == cls)[0]
        chosen = rng.choice(idx, size=n_labeled_per_class, replace=False)
        for i in chosen:
            labels_known[int(i)] = cls

    u_final, y_pred = classify(
        X, labels_known,
        k=10, sigma=0.3, m=20,
        epsilon=1.0, dt=0.01, C=1.0, mu=50.0, n_iter=500,
    )

    accuracy = np.mean(y_pred == y_true)
    print(f"Known labels used: {len(labels_known)} out of {len(y_true)} points")
    print(f"Classification accuracy: {accuracy:.3f}")

    # Visualise
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    axes[0].scatter(X[:, 0], X[:, 1], c=y_true, cmap="coolwarm", s=15)
    labeled_idx = list(labels_known.keys())
    axes[0].scatter(X[labeled_idx, 0], X[labeled_idx, 1],
                     facecolors="none", edgecolors="black", s=100, linewidths=1.5,
                     label="known labels")
    axes[0].set_title("Ground truth (known labels circled)")
    axes[0].legend()

    axes[1].scatter(X[:, 0], X[:, 1], c=y_pred, cmap="coolwarm", s=15)
    axes[1].set_title(f"Predicted classes (accuracy = {accuracy:.1%})")

    plt.tight_layout()
    plt.savefig("two_moons_result.png", dpi=150)
    print("Saved plot -> two_moons_result.png")


if __name__ == "__main__":
    main()
