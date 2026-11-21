import numpy as np
from network import (
    backward_pass,
    forward_pass,
    init_parameters,
    update_parameters,
)
from plots import (
    plot_classification_error,
    plot_mse,
    plot_weights,
)


def get_xor_data():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    Y = np.array([[0], [1], [1], [0]], dtype=float)
    return X, Y


def compute_metrics(Y, A2, E1, E2):
    # MSE (Mean Squared Error - błąd średniokwadratowy) warstwy wyjściowej
    sq_err_out = E2**2
    mse_out_samples = sq_err_out.flatten()
    mse_out_total = float(np.mean(sq_err_out))

    # MSE warstwy ukrytej
    sq_err_hid = E1**2
    mse_hid_samples = np.mean(sq_err_hid, axis=1)
    mse_hid_total = float(np.mean(sq_err_hid))

    # Błąd klasyfikacji 0-1 (próg 0.5)
    predictions = (A2 >= 0.5).astype(float)
    class_error = int(np.sum(predictions != Y))

    return {
        "mse_out_total": mse_out_total,
        "mse_out_samples": mse_out_samples,
        "mse_hid_total": mse_hid_total,
        "mse_hid_samples": mse_hid_samples,
        "class_error": class_error,
    }


def train(
    X,
    Y,
    hidden_size=3,
    learning_rate=0.5,
    max_epochs=10000,
    target_error=0.001,
    seed=42,
):
    params = init_parameters(
        input_size=2,
        hidden_size=hidden_size,
        output_size=1,
        seed=seed,
    )

    history = {
        "mse_out_total": [],
        "mse_out_samples": [],
        "mse_hid_total": [],
        "mse_hid_samples": [],
        "class_errors": [],
        "w1": [],
        "w2": [],
    }

    stopped_epoch = max_epochs

    for epoch in range(1, max_epochs + 1):
        cache = forward_pass(X, params)
        grads, errors = backward_pass(X, Y, params, cache)
        metrics = compute_metrics(Y, cache["A2"], errors["E1"], errors["E2"])

        # Zapis metryk i wag z bieżącej epoki
        history["mse_out_total"].append(metrics["mse_out_total"])
        history["mse_out_samples"].append(metrics["mse_out_samples"])
        history["mse_hid_total"].append(metrics["mse_hid_total"])
        history["mse_hid_samples"].append(metrics["mse_hid_samples"])
        history["class_errors"].append(metrics["class_error"])
        history["w1"].append(params["W1"].flatten().copy())
        history["w2"].append(params["W2"].flatten().copy())

        update_parameters(params, grads, learning_rate)

        # Wcześniejsze zakończenie uczenia po osiągnięciu zadanego błędu
        if metrics["mse_out_total"] <= target_error:
            stopped_epoch = epoch
            break

    for key in history:
        history[key] = np.array(history[key])

    return params, history, stopped_epoch


if __name__ == "__main__":
    X, Y = get_xor_data()

    # Hiperparametry do eksperymentów w sprawozdaniu
    HIDDEN_SIZE = 3
    LEARNING_RATE = 0.5
    MAX_EPOCHS = 10000
    TARGET_ERROR = 0.001
    SEED = 42

    trained_params, history, final_epoch = train(
        X=X,
        Y=Y,
        hidden_size=HIDDEN_SIZE,
        learning_rate=LEARNING_RATE,
        max_epochs=MAX_EPOCHS,
        target_error=TARGET_ERROR,
        seed=SEED,
    )

    # Podsumowanie wyników w konsoli
    final_output = forward_pass(X, trained_params)["A2"]
    print(f"Zakończono na epoce: {final_epoch} / {MAX_EPOCHS}")
    print(f"Końcowe MSE warstwy wyjściowej: {history['mse_out_total'][-1]:.6f}")
    print(f"Końcowy błąd klasyfikacji: {history['class_errors'][-1]} / 4\n")

    print("Wejście | Oczekiwane | Predykcja sieci | Po progowaniu (0-1)")
    for x_row, y_true, y_pred in zip(X, Y, final_output):
        pred_bin = int(y_pred[0] >= 0.5)
        print(
            f" {int(x_row[0])}, {int(x_row[1])}   |     {int(y_true[0])}      |"
            f"     {y_pred[0]:.4f}      |          {pred_bin}"
        )

    # Generowanie kompletu wykresów do sprawozdania
    plot_mse(
        history["mse_out_total"],
        history["mse_out_samples"],
        layer_name="warstwa wyjściowa",
        filename="mse_wyjsciowa.png",
    )
    plot_mse(
        history["mse_hid_total"],
        history["mse_hid_samples"],
        layer_name="warstwa ukryta",
        filename="mse_ukryta.png",
    )
    plot_classification_error(
        history["class_errors"],
        filename="blad_klasyfikacji.png",
    )
    plot_weights(
        history["w1"],
        layer_name="warstwa ukryta (W1)",
        filename="wagi_ukryta.png",
    )
    plot_weights(
        history["w2"],
        layer_name="warstwa wyjściowa (W2)",
        filename="wagi_wyjsciowa.png",
    )