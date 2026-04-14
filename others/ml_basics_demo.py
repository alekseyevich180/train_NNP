from __future__ import annotations

import numpy as np


def make_toy_dataset(num_samples: int = 240, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Generate a small binary classification dataset.

    Feature meaning:
    - x[:, 0]: "reaction_activity"
    - x[:, 1]: "surface_stability"
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, size=(num_samples, 2)).astype(np.float64)
    signal = 1.4 * x[:, 0] - 0.9 * x[:, 1] + 0.25 * rng.normal(size=num_samples)
    y = (signal > 0.0).astype(np.float64)
    return x, y


def train_test_split(
    x: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(x))
    rng.shuffle(indices)
    train_size = int(len(x) * train_ratio)
    train_idx = indices[:train_size]
    test_idx = indices[train_size:]
    return x[train_idx], x[test_idx], y[train_idx], y[test_idx]


def standardize_train_test(
    x_train: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std == 0] = 1.0
    return (x_train - mean) / std, (x_test - mean) / std, mean, std


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50.0, 50.0)))


def binary_cross_entropy(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    eps = 1.0e-8
    y_prob = np.clip(y_prob, eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(y_prob) + (1.0 - y_true) * np.log(1.0 - y_prob)))


def predict_proba(x: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    return sigmoid(x @ weights + bias)


def predict_label(x: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    return (predict_proba(x, weights, bias) >= 0.5).astype(np.float64)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def train_logistic_regression(
    x_train: np.ndarray,
    y_train: np.ndarray,
    learning_rate: float = 0.1,
    epochs: int = 300,
) -> tuple[np.ndarray, float, list[float]]:
    """Train logistic regression from scratch with gradient descent."""
    weights = np.zeros(x_train.shape[1], dtype=np.float64)
    bias = 0.0
    loss_history: list[float] = []

    for epoch in range(epochs):
        probs = predict_proba(x_train, weights, bias)
        error = probs - y_train

        grad_w = (x_train.T @ error) / len(x_train)
        grad_b = float(np.mean(error))

        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b

        loss = binary_cross_entropy(y_train, probs)
        loss_history.append(loss)

        if epoch % 50 == 0 or epoch == epochs - 1:
            print(f"epoch={epoch:03d} loss={loss:.4f}")

    return weights, bias, loss_history


def print_interpretation(weights: np.ndarray, bias: float, mean: np.ndarray, std: np.ndarray) -> None:
    feature_names = ["reaction_activity", "surface_stability"]
    print("\nModel interpretation")
    print("--------------------")
    for name, weight, mu, sigma in zip(feature_names, weights, mean, std, strict=True):
        direction = "positive" if weight >= 0 else "negative"
        print(
            f"{name:>18}: weight={weight:+.4f}, direction={direction}, "
            f"train_mean={mu:+.4f}, train_std={sigma:.4f}"
        )
    print(f"{'bias':>18}: value={bias:+.4f}")


def explain_ml_concepts() -> None:
    print("\nBasic ML concepts")
    print("-----------------")
    print("1. Dataset: each row is one sample and each column is one feature.")
    print("2. Label: 0 or 1 tells the model which class the sample belongs to.")
    print("3. Standardization: keep features on similar scales before training.")
    print("4. Model: logistic regression maps features to a probability.")
    print("5. Loss: binary cross-entropy measures prediction error.")
    print("6. Training: gradient descent updates parameters to reduce loss.")
    print("7. Evaluation: test accuracy checks whether the model generalizes.")


def main() -> None:
    print("Machine learning basics demo")
    print("============================")

    x, y = make_toy_dataset()
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    x_train_std, x_test_std, mean, std = standardize_train_test(x_train, x_test)

    print(f"all samples      : {len(x)}")
    print(f"training samples : {len(x_train)}")
    print(f"test samples     : {len(x_test)}")
    print(f"positive ratio   : {float(y.mean()):.3f}")

    weights, bias, _ = train_logistic_regression(
        x_train_std,
        y_train,
        learning_rate=0.15,
        epochs=300,
    )

    train_prob = predict_proba(x_train_std, weights, bias)
    test_prob = predict_proba(x_test_std, weights, bias)
    train_pred = predict_label(x_train_std, weights, bias)
    test_pred = predict_label(x_test_std, weights, bias)

    print("\nFinal metrics")
    print("-------------")
    print(f"train loss     : {binary_cross_entropy(y_train, train_prob):.4f}")
    print(f"test loss      : {binary_cross_entropy(y_test, test_prob):.4f}")
    print(f"train accuracy : {accuracy(y_train, train_pred):.4f}")
    print(f"test accuracy  : {accuracy(y_test, test_pred):.4f}")

    print_interpretation(weights, bias, mean, std)
    explain_ml_concepts()


if __name__ == "__main__":
    main()
