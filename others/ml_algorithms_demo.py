from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np


@dataclass
class SplitDataset:
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray


def train_test_split(
    x: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> SplitDataset:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(x))
    rng.shuffle(indices)
    train_size = int(len(x) * train_ratio)
    train_idx = indices[:train_size]
    test_idx = indices[train_size:]
    return SplitDataset(
        x_train=x[train_idx],
        x_test=x[test_idx],
        y_train=y[train_idx],
        y_test=y[test_idx],
    )


def standardize_train_test(
    x_train: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std == 0.0] = 1.0
    return (x_train - mean) / std, (x_test - mean) / std, mean, std


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def make_knn_dataset(num_samples: int = 240, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    class0 = rng.normal(loc=(-1.4, -1.0), scale=0.65, size=(num_samples // 2, 2))
    class1 = rng.normal(loc=(1.3, 1.2), scale=0.65, size=(num_samples // 2, 2))
    x = np.vstack([class0, class1]).astype(np.float64)
    y = np.concatenate([np.zeros(len(class0)), np.ones(len(class1))]).astype(np.int64)
    return x, y


def knn_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_query: np.ndarray,
    k: int = 5,
) -> np.ndarray:
    distances = np.linalg.norm(x_query[:, None, :] - x_train[None, :, :], axis=2)
    nearest_idx = np.argsort(distances, axis=1)[:, :k]
    nearest_labels = y_train[nearest_idx]
    return (nearest_labels.mean(axis=1) >= 0.5).astype(np.int64)


def demo_knn() -> None:
    print("K-Nearest Neighbors demo")
    print("========================")
    x, y = make_knn_dataset()
    split = train_test_split(x, y)
    x_train, x_test, mean, std = standardize_train_test(split.x_train, split.x_test)
    predictions = knn_predict(x_train, split.y_train, x_test, k=5)

    print(f"training samples : {len(x_train)}")
    print(f"test samples     : {len(x_test)}")
    print(f"test accuracy    : {accuracy(split.y_test, predictions):.4f}")
    print("example predictions:")
    for idx in range(5):
        print(f"  sample={idx:02d}, pred={predictions[idx]}, truth={split.y_test[idx]}")
    print(f"standardization mean={mean.round(3)}, std={std.round(3)}")


def make_kmeans_dataset(seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cluster_a = rng.normal(loc=(-3.5, 1.5), scale=0.55, size=(70, 2))
    cluster_b = rng.normal(loc=(0.8, -2.5), scale=0.60, size=(70, 2))
    cluster_c = rng.normal(loc=(3.5, 2.8), scale=0.50, size=(70, 2))
    return np.vstack([cluster_a, cluster_b, cluster_c]).astype(np.float64)


def kmeans_fit(
    x: np.ndarray,
    k: int = 3,
    max_iter: int = 100,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centers = x[rng.choice(len(x), size=k, replace=False)].copy()

    for _ in range(max_iter):
        distances = np.linalg.norm(x[:, None, :] - centers[None, :, :], axis=2)
        labels = np.argmin(distances, axis=1)
        new_centers = centers.copy()

        for cluster_id in range(k):
            members = x[labels == cluster_id]
            if len(members) > 0:
                new_centers[cluster_id] = members.mean(axis=0)

        if np.allclose(new_centers, centers):
            centers = new_centers
            break
        centers = new_centers

    distances = np.linalg.norm(x[:, None, :] - centers[None, :, :], axis=2)
    labels = np.argmin(distances, axis=1)
    return centers, labels


def demo_kmeans() -> None:
    print("K-Means demo")
    print("============")
    x = make_kmeans_dataset()
    centers, labels = kmeans_fit(x, k=3)

    print(f"num_samples : {len(x)}")
    for cluster_id in range(3):
        count = int(np.sum(labels == cluster_id))
        center = np.round(centers[cluster_id], 3)
        print(f"cluster={cluster_id}, count={count}, center={center}")


def make_regression_dataset(num_samples: int = 240, seed: int = 24) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3.0, 3.0, size=(num_samples, 2)).astype(np.float64)
    noise = rng.normal(loc=0.0, scale=0.6, size=num_samples)
    y = 2.6 * x[:, 0] - 1.8 * x[:, 1] + 4.2 + noise
    return x, y.astype(np.float64)


def train_linear_regression(
    x_train: np.ndarray,
    y_train: np.ndarray,
    learning_rate: float = 0.05,
    epochs: int = 400,
) -> tuple[np.ndarray, float]:
    weights = np.zeros(x_train.shape[1], dtype=np.float64)
    bias = 0.0

    for epoch in range(epochs):
        preds = x_train @ weights + bias
        error = preds - y_train
        grad_w = (x_train.T @ error) / len(x_train)
        grad_b = float(np.mean(error))

        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b

        if epoch % 100 == 0 or epoch == epochs - 1:
            loss = mean_squared_error(y_train, preds)
            print(f"epoch={epoch:03d} mse={loss:.4f}")

    return weights, bias


def demo_regression() -> None:
    print("Linear Regression demo")
    print("======================")
    x, y = make_regression_dataset()
    split = train_test_split(x, y)
    x_train, x_test, mean, std = standardize_train_test(split.x_train, split.x_test)
    weights, bias = train_linear_regression(x_train, split.y_train)

    train_pred = x_train @ weights + bias
    test_pred = x_test @ weights + bias

    print("\nFinal metrics")
    print("-------------")
    print(f"train mse : {mean_squared_error(split.y_train, train_pred):.4f}")
    print(f"test mse  : {mean_squared_error(split.y_test, test_pred):.4f}")
    print(f"weights   : {np.round(weights, 4)}")
    print(f"bias      : {bias:.4f}")
    print(f"feature mean={mean.round(3)}, std={std.round(3)}")


def make_perceptron_dataset(num_samples: int = 220, seed: int = 11) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, size=(num_samples, 2)).astype(np.float64)
    margin = 1.2 * x[:, 0] - 0.7 * x[:, 1] + 0.1
    y = np.where(margin >= 0.0, 1, -1).astype(np.int64)
    return x, y


def train_perceptron(
    x_train: np.ndarray,
    y_train: np.ndarray,
    learning_rate: float = 0.1,
    epochs: int = 30,
) -> tuple[np.ndarray, float]:
    weights = np.zeros(x_train.shape[1], dtype=np.float64)
    bias = 0.0

    for epoch in range(epochs):
        errors = 0
        for xi, yi in zip(x_train, y_train, strict=True):
            activation = float(np.dot(xi, weights) + bias)
            if yi * activation <= 0.0:
                weights += learning_rate * yi * xi
                bias += learning_rate * yi
                errors += 1

        print(f"epoch={epoch:02d} mistakes={errors}")
        if errors == 0:
            break

    return weights, bias


def perceptron_predict(x: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    return np.where(x @ weights + bias >= 0.0, 1, -1).astype(np.int64)


def demo_perceptron() -> None:
    print("Perceptron demo")
    print("===============")
    x, y = make_perceptron_dataset()
    split = train_test_split(x, y)
    x_train, x_test, mean, std = standardize_train_test(split.x_train, split.x_test)
    weights, bias = train_perceptron(x_train, split.y_train)
    predictions = perceptron_predict(x_test, weights, bias)

    print("\nFinal metrics")
    print("-------------")
    print(f"test accuracy : {accuracy(split.y_test, predictions):.4f}")
    print(f"weights       : {np.round(weights, 4)}")
    print(f"bias          : {bias:.4f}")
    print(f"feature mean={mean.round(3)}, std={std.round(3)}")


def run_all() -> None:
    print("Machine learning algorithm demos")
    print("================================")
    print("Included methods from the image:")
    print("1. K-nearest neighbors")
    print("2. K-means clustering")
    print("3. Linear regression")
    print("4. Perceptron")
    print("\nNote: '常用算法' looks like a section title rather than one specific algorithm.")
    print()
    demo_knn()
    print()
    demo_kmeans()
    print()
    demo_regression()
    print()
    demo_perceptron()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run simple machine-learning algorithm demos implemented with numpy."
    )
    parser.add_argument(
        "--method",
        choices=["all", "knn", "kmeans", "regression", "perceptron"],
        default="all",
        help="Choose which method demo to run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.method == "all":
        run_all()
    elif args.method == "knn":
        demo_knn()
    elif args.method == "kmeans":
        demo_kmeans()
    elif args.method == "regression":
        demo_regression()
    elif args.method == "perceptron":
        demo_perceptron()


if __name__ == "__main__":
    main()
