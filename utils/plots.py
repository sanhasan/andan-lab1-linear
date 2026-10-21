from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def format_log_value(value):
    if value == 0:
        return "0"

    exponent = np.log10(value)
    if np.isclose(exponent, round(exponent)):
        exponent = int(round(exponent))
    else:
        exponent = f"{exponent:.2f}".rstrip("0").rstrip(".")
    return rf"$10^{{{exponent}}}$"


def save_figure(figure, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, bbox_inches="tight")
    return figure


def plot_ridge_search(results, path):
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.semilogx(results["alphas"], results["scores"], marker="o")
    axis.axvline(
        results["best_alpha"],
        color="tab:red",
        linestyle="--",
        label=f'лучшее α = {results["best_alpha"]:.3g}',
    )
    axis.set_xlabel("α")
    axis.set_ylabel("Test accuracy")
    axis.set_title("Подбор коэффициента Ridge-регуляризации")
    axis.grid(alpha=0.3)
    axis.legend()
    return save_figure(figure, path)


def plot_gradient_search(results, path):
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    for loss, scores in zip(results["losses"], results["loss_scores"]):
        axes[0].semilogx(
            results["learning_rates"], scores, marker="o", label=loss
        )
    axes[0].set_xlabel("Скорость обучения η")
    axes[0].set_ylabel("Test accuracy")
    axes[0].set_title("Выбор риска и скорости обучения")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    image = axes[1].imshow(
        results["regularization_scores"],
        origin="lower",
        aspect="auto",
        cmap="viridis",
        vmin=results["regularization_scores"].min(),
        vmax=results["regularization_scores"].max(),
    )
    axes[1].set_xticks(range(len(results["l2_values"])))
    axes[1].set_xticklabels(
        [format_log_value(value) for value in results["l2_values"]]
    )
    axes[1].set_yticks(range(len(results["l1_values"])))
    axes[1].set_yticklabels(
        [format_log_value(value) for value in results["l1_values"]]
    )
    axes[1].set_xlabel("λ₂")
    axes[1].set_ylabel("λ₁")
    axes[1].set_title("Подбор Elastic Net")
    figure.colorbar(image, ax=axes[1], label="Test accuracy")

    figure.tight_layout()
    return save_figure(figure, path)


def plot_train_risk(iterations, risk, window, path):
    window = min(window, len(risk))
    smoothed = np.convolve(risk, np.ones(window) / window, mode="valid")
    smoothed_iterations = iterations[window - 1 :]

    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(iterations, risk, alpha=0.3, label="эмпирический риск")
    axis.plot(
        smoothed_iterations,
        smoothed,
        linewidth=2,
        label=f"скользящее среднее, окно {window}",
    )
    axis.set_xlabel("Итерация")
    axis.set_ylabel("Эмпирический риск")
    axis.set_title("Кривая обучения на тренировочной выборке")
    axis.grid(alpha=0.3)
    axis.legend()
    return save_figure(figure, path)


def plot_test_accuracy(results, path):
    confidence = f"{results['confidence']:.0%}"
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(
        results["iterations"],
        results["gradient_mean"],
        color="tab:blue",
        label="градиентный классификатор",
    )
    axis.fill_between(
        results["iterations"],
        results["gradient_lower"],
        results["gradient_upper"],
        color="tab:blue",
        alpha=0.2,
        label=f"{confidence} доверительный интервал градиентной модели",
    )
    axis.axhline(
        results["ridge_mean"],
        color="tab:red",
        linestyle="--",
        label=f'Ridge: {results["ridge_mean"]:.3f}',
    )
    axis.axhspan(
        results["ridge_lower"],
        results["ridge_upper"],
        color="tab:red",
        alpha=0.1,
        label=f"{confidence} доверительный интервал Ridge",
    )
    axis.set_xlabel("Итерация")
    axis.set_ylabel("Test accuracy")
    axis.set_title("Кривая обучения с доверительным интервалом")
    axis.set_ylim(0.5, 1.0)
    axis.grid(alpha=0.3)
    axis.legend()
    return save_figure(figure, path)


def plot_coefficient_path(results, features_count, path, random_state):
    coefficients = results["coefficients"]
    generator = np.random.default_rng(random_state)
    selected = np.sort(
        generator.choice(
            coefficients.shape[1],
            size=features_count,
            replace=False,
        )
    )
    name = results["regularization"].upper()

    figure, axis = plt.subplots(figsize=(9, 5))
    for feature in selected:
        axis.semilogx(
            results["values"],
            coefficients[:, feature],
            label=f"w{feature}",
        )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xlabel(f"Коэффициент {name}-регуляризации")
    axis.set_ylabel("Коэффициент модели")
    axis.set_title(f"Зависимость коэффициентов модели от {name}")
    axis.grid(alpha=0.3)
    axis.legend(ncol=2, fontsize=8)
    return save_figure(figure, path)
