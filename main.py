import math
import random
import matplotlib.pyplot as plt


def sigmoid(x):
    x = max(-500, min(500, x))
    return 1.0 / (1.0 + math.exp(-x))


def pochodna_sigmoid(a):
    return a * (1.0 - a)


class SiecNeuronowa:
    def __init__(self, wejscia, ukryte, wyjscia, lr=0.8, momentum=0.9, batch_size=1, seed=42):
        random.seed(seed)
        self.wejscia = wejscia
        self.ukryte = ukryte
        self.wyjscia = wyjscia
        self.lr = lr
        self.momentum = momentum
        self.batch_size = batch_size

        # Pamięć prędkości (dla WSZYSTKICH wag i biasów)
        self.vW1 = [[0.0 for _ in range(ukryte)] for _ in range(wejscia)]
        self.vb1 = [0.0 for _ in range(ukryte)]
        self.vW2 = [[0.0 for _ in range(wyjscia)] for _ in range(ukryte)]
        self.vb2 = [0.0 for _ in range(wyjscia)]

        # Wagi i biasy początkowe
        self.W1 = [[random.uniform(-1, 1) for _ in range(ukryte)] for _ in range(wejscia)]
        self.b1 = [random.uniform(-1, 1) for _ in range(ukryte)]

        self.W2 = [[random.uniform(-1, 1) for _ in range(wyjscia)] for _ in range(ukryte)]
        self.b2 = [random.uniform(-1, 1) for _ in range(wyjscia)]

        # Listy do zbierania danych na potrzeby wykresów do sprawozdania
        self.historia_mse_out = []
        self.historia_mse_h = []
        self.historia_bledu_01 = []
        self.historia_W1 = []
        self.historia_W2 = []

    def w_przod(self, x):
        h = []
        for j in range(self.ukryte):
            suma = self.b1[j]
            for i in range(self.wejscia):
                suma += x[i] * self.W1[i][j]
            h.append(sigmoid(suma))

        out = []
        for k in range(self.wyjscia):
            suma = self.b2[k]
            for j in range(self.ukryte):
                suma += h[j] * self.W2[j][k]
            out.append(sigmoid(suma))

        return h, out

    def trenuj(self, X, Y, max_epok=5000, docelowy_mse=0.001, cicho=False):
        n_probek = len(X)

        for epoka in range(1, max_epok + 1):
            suma_mse_out = 0.0
            suma_mse_h = 0.0
            zle_klasyfikacje = 0

            # Dzielimy zbiór danych na paczki o rozmiarze self.batch_size
            for start in range(0, n_probek, self.batch_size):
                X_batch = X[start : start + self.batch_size]
                Y_batch = Y[start : start + self.batch_size]
                aktualny_rozmiar_paczki = len(X_batch)

                # Zerujemy akumulatory ("wiaderka") na poprawki z bieżącej paczki
                akum_W1 = [[0.0 for _ in range(self.ukryte)] for _ in range(self.wejscia)]
                akum_b1 = [0.0 for _ in range(self.ukryte)]
                akum_W2 = [[0.0 for _ in range(self.wyjscia)] for _ in range(self.ukryte)]
                akum_b2 = [0.0 for _ in range(self.wyjscia)]

                # 1. Zbieramy poprawki dla wszystkich przykładów z paczki
                for x, y in zip(X_batch, Y_batch):
                    # A. Propagacja w przód
                    h, out = self.w_przod(x)

                    # B. Błąd na wyjściu (delta_k)
                    korekta_out = []
                    for k in range(self.wyjscia):
                        blad = y[k] - out[k]
                        suma_mse_out += blad ** 2
                        korekta_out.append(blad * pochodna_sigmoid(out[k]))

                        predykcja_binarna = 1 if out[k] >= 0.5 else 0
                        if predykcja_binarna != y[k]:
                            zle_klasyfikacje += 1

                    # C. Błąd w warstwie ukrytej (delta_j)
                    korekta_h = []
                    for j in range(self.ukryte):
                        blad_ukryty = 0.0
                        for k in range(self.wyjscia):
                            blad_ukryty += korekta_out[k] * self.W2[j][k]
                        suma_mse_h += blad_ukryty ** 2
                        korekta_h.append(blad_ukryty * pochodna_sigmoid(h[j]))

                    # D. Wrzucamy poprawki do akumulatorów (jeszcze NIE zmieniamy wag!)
                    for j in range(self.ukryte):
                        for k in range(self.wyjscia):
                            akum_W2[j][k] += korekta_out[k] * h[j]
                    for k in range(self.wyjscia):
                        akum_b2[k] += korekta_out[k]

                    for i in range(self.wejscia):
                        for j in range(self.ukryte):
                            akum_W1[i][j] += korekta_h[j] * x[i]
                    for j in range(self.ukryte):
                        akum_b1[j] += korekta_h[j]

                # 2. Koniec paczki -> uśredniamy poprawki i aktualizujemy WSZYSTKIE wagi z momentum
                for j in range(self.ukryte):
                    for k in range(self.wyjscia):
                        srednia_poprawka = akum_W2[j][k] / aktualny_rozmiar_paczki
                        zmiana = (self.lr * srednia_poprawka) + (self.momentum * self.vW2[j][k])
                        self.vW2[j][k] = zmiana
                        self.W2[j][k] += zmiana

                for k in range(self.wyjscia):
                    srednia_poprawka = akum_b2[k] / aktualny_rozmiar_paczki
                    zmiana = (self.lr * srednia_poprawka) + (self.momentum * self.vb2[k])
                    self.vb2[k] = zmiana
                    self.b2[k] += zmiana

                for i in range(self.wejscia):
                    for j in range(self.ukryte):
                        srednia_poprawka = akum_W1[i][j] / aktualny_rozmiar_paczki
                        zmiana = (self.lr * srednia_poprawka) + (self.momentum * self.vW1[i][j])
                        self.vW1[i][j] = zmiana
                        self.W1[i][j] += zmiana

                for j in range(self.ukryte):
                    srednia_poprawka = akum_b1[j] / aktualny_rozmiar_paczki
                    zmiana = (self.lr * srednia_poprawka) + (self.momentum * self.vb1[j])
                    self.vb1[j] = zmiana
                    self.b1[j] += zmiana

            # Zapisujemy metryki po zakończeniu pełnej epoki
            mse_out = suma_mse_out / (n_probek * self.wyjscia)
            mse_h = suma_mse_h / (n_probek * self.ukryte)
            blad_01 = zle_klasyfikacje / (n_probek * self.wyjscia)

            self.historia_mse_out.append(mse_out)
            self.historia_mse_h.append(mse_h)
            self.historia_bledu_01.append(blad_01)
            self.historia_W1.append([waga for wiersz in self.W1 for waga in wiersz])
            self.historia_W2.append([waga for wiersz in self.W2 for waga in wiersz])

            # Warunek wczesnego zatrzymania (Early Stopping)
            if mse_out <= docelowy_mse:
                if not cicho:
                    print(f"[Early Stopping] Osiągnięto cel w epoce {epoka}! (MSE = {mse_out:.6f})")
                return epoka

        if not cicho:
            print(f"Zakończono po limicie {max_epok} epok (Końcowe MSE = {mse_out:.6f})")
        return max_epok

    def rysuj_wykresy(self, plik_wyjsciowy="wykresy_xor.png"):
        epoki = range(1, len(self.historia_mse_out) + 1)
        fig, axs = plt.subplots(2, 2, figsize=(13, 9))
        fig.suptitle(
            f"Sieć XOR (2->{self.ukryte}->1 | lr={self.lr}, mom={self.momentum}, batch={self.batch_size})",
            fontsize=13,
        )

        # 1. Wykres błędu MSE w obu warstwach
        axs[0, 0].plot(epoki, self.historia_mse_out, label="Warstwa wyjściowa (MSE)", color="tab:blue")
        axs[0, 0].plot(epoki, self.historia_mse_h, label="Warstwa ukryta (MSE)", color="tab:orange", linestyle="--")
        axs[0, 0].set_title("Błąd średniokwadratowy (MSE)")
        axs[0, 0].set_xlabel("Epoka")
        axs[0, 0].set_ylabel("Wartość MSE")
        axs[0, 0].legend()
        axs[0, 0].grid(True, alpha=0.3)

        # 2. Wykres błędu klasyfikacji 0-1
        axs[0, 1].plot(epoki, self.historia_bledu_01, color="tab:red")
        axs[0, 1].set_title("Błąd klasyfikacji (0–1)")
        axs[0, 1].set_xlabel("Epoka")
        axs[0, 1].set_ylabel("Odsetek błędnych odpowiedzi")
        axs[0, 1].set_ylim(-0.05, 1.05)
        axs[0, 1].grid(True, alpha=0.3)

        # 3. Zmiana wag w warstwie 1 (ukrytej)
        for idx in range(len(self.historia_W1[0])):
            wagi_linia = [epoka_w[idx] for epoka_w in self.historia_W1]
            axs[1, 0].plot(epoki, wagi_linia, label=f"w1_{idx}")
        axs[1, 0].set_title("Ewolucja wag w warstwie 1 (ukrytej)")
        axs[1, 0].set_xlabel("Epoka")
        axs[1, 0].set_ylabel("Wartość wagi")
        axs[1, 0].legend(loc="best", fontsize=8)
        axs[1, 0].grid(True, alpha=0.3)

        # 4. Zmiana wag w warstwie 2 (wyjściowej)
        for idx in range(len(self.historia_W2[0])):
            wagi_linia = [epoka_w[idx] for epoka_w in self.historia_W2]
            axs[1, 1].plot(epoki, wagi_linia, label=f"w2_{idx}")
        axs[1, 1].set_title("Ewolucja wag w warstwie 2 (wyjściowej)")
        axs[1, 1].set_xlabel("Epoka")
        axs[1, 1].set_ylabel("Wartość wagi")
        axs[1, 1].legend(loc="best", fontsize=8)
        axs[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(plik_wyjsciowy, dpi=200)
        print(f"Zapisano wykresy do pliku: {plik_wyjsciowy}")
        plt.show()


# ==========================================
# URUCHOMIENIE EKSPERYMENTU DLA BRAMKI XOR
# ==========================================
if __name__ == "__main__":
    X_xor = [[0, 0], [0, 1], [1, 0], [1, 1]]
    Y_xor = [[0],    [1],    [1],    [0]]

    # 1. Pojedynczy przebieg z momentum=0.9 i batch_size=2 (mini-batch)
    siec = SiecNeuronowa(wejscia=2, ukryte=2, wyjscia=1, lr=0.8, momentum=0.9, batch_size=2, seed=42)
    siec.trenuj(X_xor, Y_xor, max_epok=5000, docelowy_mse=0.001)

    print("\nWeryfikacja po treningu:")
    for x, y in zip(X_xor, Y_xor):
        _, out = siec.w_przod(x)
        print(f"Wejście {x} | Oczekiwane: {y[0]} | Sieć: {out[0]:.4f}")

    siec.rysuj_wykresy("wykresy_xor_batch2.png")

    # 2. Porównanie SGD (batch=1), Mini-Batch (batch=2) oraz Full-Batch (batch=4)
    plt.figure(figsize=(9, 5))
    for bs, nazwa in [(1, "SGD (batch_size=1)"), (2, "Mini-Batch (batch_size=2)"), (4, "Full-Batch (batch_size=4)")]:
        s = SiecNeuronowa(wejscia=2, ukryte=2, wyjscia=1, lr=0.8, momentum=0.9, batch_size=bs, seed=42)
        epoki = s.trenuj(X_xor, Y_xor, max_epok=5000, docelowy_mse=0.001, cicho=True)
        plt.plot(range(1, len(s.historia_mse_out) + 1), s.historia_mse_out, label=f"{nazwa} -> {epoki} epok")

    plt.title("Porównanie trybów uczenia: SGD vs Mini-Batch vs Full-Batch")
    plt.xlabel("Epoka")
    plt.ylabel("MSE")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("porownanie_batch_size.png", dpi=200)
    plt.show()