import matplotlib.pyplot as plt
import numpy as np


def plot_mse(mse_total, mse_samples, layer_name, filename):
    plt.figure(figsize=(10, 6))
    labels = ["(0, 0)", "(0, 1)", "(1, 0)", "(1, 1)"]

    for i in range(4):
        plt.plot(
            mse_samples[:, i],
            linestyle="--",
            alpha=0.75,
            label=f"Przykład {labels[i]}",
        )

    plt.plot(
        mse_total,
        color="black",
        linewidth=2.2,
        label="Cały zbiór uczący (średnia)",
    )

    plt.title(f"Błąd MSE (błąd średniokwadratowy) – {layer_name}")
    plt.xlabel("Epoka")
    plt.ylabel("MSE")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_classification_error(class_errors, filename):
    plt.figure(figsize=(10, 5))
    plt.plot(class_errors, color="crimson", linewidth=1.8)

    plt.title("Błąd klasyfikacji (0-1) na przestrzeni epok")
    plt.xlabel("Epoka")
    plt.ylabel("Liczba błędnie sklasyfikowanych przykładów")
    plt.yticks([0, 1, 2, 3, 4])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_weights(weights_history, layer_name, filename):
    plt.figure(figsize=(10, 6))
    num_weights = weights_history.shape[1]

    for i in range(num_weights):
        plt.plot(weights_history[:, i], label=f"Waga {i + 1}")

    plt.title(f"Zmiana wartości wag w czasie uczenia - {layer_name}")
    plt.xlabel("Epoka")
    plt.ylabel("Wartość wagi")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", ncol=2)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_learning_rate(lr_history, filename="learning_rate.png"):
    epochs = np.arange(1, len(lr_history) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, lr_history, color="tab:purple", linewidth=2, label="Współczynnik uczenia (alpha)")
    plt.title("Zmiana adaptacyjnego współczynnika uczenia (learning rate) w czasie")
    plt.xlabel("Epoka")
    plt.ylabel("Learning rate (alpha)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()