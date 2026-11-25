import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


def _downsample(arr, max_points=2500):
    """Ogranicza liczbę rysowanych punktów do max_points, zachowując realne numery epok."""
    n = len(arr)
    step = max(1, n // max_points)
    epochs = np.arange(1, n + 1)[::step]
    return epochs, arr[::step]


def _ensure_dir(filename):
    """Tworzy katalog docelowy (np. plots/), jeśli jeszcze nie istnieje."""
    directory = os.path.dirname(filename)
    if directory:
        os.makedirs(directory, exist_ok=True)


def plot_mse(mse_total, mse_samples, layer_name, filename):
    _ensure_dir(filename)
    epochs, mse_total_ds = _downsample(mse_total)
    _, mse_samples_ds = _downsample(mse_samples)

    plt.figure(figsize=(10, 6))
    num_samples = mse_samples_ds.shape[1]

    if num_samples <= 8:
        for i in range(num_samples):
            plt.plot(
                epochs,
                mse_samples_ds[:, i],
                linestyle="--",
                alpha=0.75,
                label=f"Przykład {i + 1}",
            )
    else:
        q25 = np.percentile(mse_samples_ds, 25, axis=1)
        q75 = np.percentile(mse_samples_ds, 75, axis=1)
        max_err = np.max(mse_samples_ds, axis=1)

        plt.fill_between(
            epochs,
            q25,
            q75,
            color="tab:blue",
            alpha=0.25,
            label="Rozstęp ćwiartkowy (25%–75% próbek)",
        )
        plt.plot(
            epochs,
            max_err,
            color="tab:orange",
            linestyle="--",
            linewidth=1.1,
            alpha=0.75,
            label="Maksymalny błąd pojedynczej próbki",
        )

    plt.plot(
        epochs,
        mse_total_ds,
        color="black",
        linewidth=2.2,
        label="Cały zbiór uczący (średnia MSE)",
    )

    plt.title(f"Błąd MSE (błąd średniokwadratowy) – {layer_name}")
    plt.xlabel("Epoka")
    plt.ylabel("MSE")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_classification_error(class_errors, filename):
    _ensure_dir(filename)
    epochs, errors_ds = _downsample(class_errors)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, errors_ds, color="crimson", linewidth=1.8)

    plt.title("Błąd klasyfikacji (0-1) na przestrzeni epok")
    plt.xlabel("Epoka")
    plt.ylabel("Liczba błędnie sklasyfikowanych przykładów")
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.ylim(bottom=0)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_weights(weights_history, layer_name, filename, max_lines=30):
    _ensure_dir(filename)
    epochs, w_ds = _downsample(weights_history)
    num_weights = w_ds.shape[1]

    plt.figure(figsize=(10, 6))

    if num_weights <= max_lines:
        for i in range(num_weights):
            plt.plot(epochs, w_ds[:, i], linewidth=1.2, label=f"Waga {i + 1}")
        title = f"Zmiana wartości wag w czasie uczenia – {layer_name}"
        if num_weights <= 15:
            plt.legend(loc="best", ncol=2)
    else:
        indices = np.linspace(0, num_weights - 1, max_lines, dtype=int)
        for idx in indices:
            plt.plot(epochs, w_ds[:, idx], linewidth=1.0, alpha=0.7)
        title = (
            f"Zmiana wartości wag w czasie uczenia – {layer_name} "
            f"(próbka {max_lines} z {num_weights} wag)"
        )

    plt.title(title)
    plt.xlabel("Epoka")
    plt.ylabel("Wartość wagi")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_learning_rate(lr_history, filename="plots/learning_rate.png"):
    _ensure_dir(filename)
    epochs, lr_ds = _downsample(lr_history)

    plt.figure(figsize=(8, 5))
    plt.plot(
        epochs,
        lr_ds,
        color="tab:purple",
        linewidth=2,
        label="Współczynnik uczenia (alpha)",
    )
    plt.title("Zmiana adaptacyjnego współczynnika uczenia (learning rate) w czasie")
    plt.xlabel("Epoka")
    plt.ylabel("Learning rate (alpha)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def generate_plots(history):
    plot_mse(
        history["mse_out_total"],
        history["mse_out_samples"],
        layer_name="warstwa wyjściowa",
        filename="plots/mse_wyjsciowa.png",
    )
    plot_mse(
        history["mse_hid_total"],
        history["mse_hid_samples"],
        layer_name="warstwa ukryta",
        filename="plots/mse_ukryta.png",
    )
    plot_classification_error(
        history["class_errors"],
        filename="plots/blad_klasyfikacji.png",
    )
    plot_weights(
        history["w1"],
        layer_name="warstwa ukryta (W1)",
        filename="plots/wagi_ukryta.png",
    )
    plot_weights(
        history["w2"],
        layer_name="warstwa wyjściowa (W2)",
        filename="plots/wagi_wyjsciowa.png",
    )
    plot_learning_rate(
        history["lr"],
        filename="plots/learning_rate.png",
    )