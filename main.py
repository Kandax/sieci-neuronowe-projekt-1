import math
import random

def sigmoid(x):
    # Zabezpieczenie przed przepełnieniem przy bardzo dużych/małych liczbach
    x = max(-500, min(500, x))
    return 1.0 / (1.0 + math.exp(-x))

def pochodna_sigmoid(a):
    return a * (1.0 - a)


class SiecNeuronowa:
    def __init__(self, wejscia, ukryte, wyjscia, lr=0.5, seed=42):
        random.seed(seed)
        self.wejscia = wejscia
        self.ukryte = ukryte
        self.wyjscia = wyjscia
        self.lr = lr

        # Tworzymy tablice wag i biasów o dowolnych wymiarach za pomocą pętli:
        # W1: lista list [wejscia x ukryte]
        self.W1 = [[random.uniform(-1, 1) for _ in range(ukryte)] for _ in range(wejscia)]
        self.b1 = [random.uniform(-1, 1) for _ in range(ukryte)]

        # W2: lista list [ukryte x wyjscia]
        self.W2 = [[random.uniform(-1, 1) for _ in range(wyjscia)] for _ in range(ukryte)]
        self.b2 = [random.uniform(-1, 1) for _ in range(wyjscia)]

    def w_przod(self, x):
        # 1. Obliczamy warstwę ukrytą (dla dowolnej liczby neuronów ukrytych)
        h = []
        for j in range(self.ukryte):
            suma = self.b1[j]
            for i in range(self.wejscia):
                suma += x[i] * self.W1[i][j]
            h.append(sigmoid(suma))

        # 2. Obliczamy warstwę wyjściową (dla dowolnej liczby neuronów wyjściowych)
        out = []
        for k in range(self.wyjscia):
            suma = self.b2[k]
            for j in range(self.ukryte):
                suma += h[j] * self.W2[j][k]
            out.append(sigmoid(suma))

        return h, out

    def trenuj(self, X, Y, max_epok=10000, docelowy_mse=0.001):
        for epoka in range(1, max_epok + 1):
            suma_bledow_epoki = 0.0

            for x, y in zip(X, Y):
                # --- A. Propagacja w przód ---
                h, out = self.w_przod(x)

                # --- B. Błąd i korekta na wyjściu ---
                korekta_out = []
                for k in range(self.wyjscia):
                    blad = y[k] - out[k]
                    suma_bledow_epoki += blad ** 2
                    korekta_out.append(blad * pochodna_sigmoid(out[k]))

                # --- C. Błąd i korekta w warstwie ukrytej ---
                korekta_h = []
                for j in range(self.ukryte):
                    blad_ukryty = 0.0
                    for k in range(self.wyjscia):
                        blad_ukryty += korekta_out[k] * self.W2[j][k]
                    korekta_h.append(blad_ukryty * pochodna_sigmoid(h[j]))

                # --- D. Aktualizacja wag i biasów ---
                # Warstwa 2 (wyjściowa)
                for j in range(self.ukryte):
                    for k in range(self.wyjscia):
                        self.W2[j][k] += self.lr * korekta_out[k] * h[j]
                for k in range(self.wyjscia):
                    self.b2[k] += self.lr * korekta_out[k]

                # Warstwa 1 (ukryta)
                for i in range(self.wejscia):
                    for j in range(self.ukryte):
                        self.W1[i][j] += self.lr * korekta_h[j] * x[i]
                for j in range(self.ukryte):
                    self.b1[j] += self.lr * korekta_h[j]

            # Liczymy średni błąd MSE po całej epoce (Early Stopping)
            mse = suma_bledow_epoki / (len(X) * self.wyjscia)
            if mse <= docelowy_mse:
                print(f"Zatrzymano w epoce {epoka} (MSE = {mse:.6f})")
                return

        print(f"Osiągnięto limit {max_epok} epok (Końcowe MSE = {mse:.6f})")


# ========================================================
# PRZYKŁAD UŻYCIA 1: Problem XOR (2 wejścia, 3 ukryte, 1 wyjście)
# ========================================================
X_xor = [[0, 0], [0, 1], [1, 0], [1, 1]]
Y_xor = [[0],    [1],    [1],    [0]]

siec = SiecNeuronowa(wejscia=2, ukryte=3, wyjscia=1, lr=0.8)
siec.trenuj(X_xor, Y_xor, max_epok=10000, docelowy_mse=0.001)

print("\nTest XOR:")
for x in X_xor:
    _, wynik = siec.w_przod(x)
    print(f"Wejście {x} -> {round(wynik[0], 3)}")


# ========================================================
# PRZYKŁAD UŻYCIA 2: Ten sam kod, ale 3 wejścia, 5 ukrytych i 2 wyjścia!
# (np. jednoczesna nauka bramki AND na 1. wyjściu oraz OR na 2. wyjściu)
# ========================================================
X_nowe = [
    [0, 0, 0],
    [0, 1, 0],
    [1, 0, 1],
    [1, 1, 1]
]
# Wyjście 1: czy wszystkie są 1 (AND), Wyjście 2: czy którakolwiek to 1 (OR)
Y_nowe = [
    [0, 0],
    [0, 1],
    [0, 1],
    [1, 1]
]

siec2 = SiecNeuronowa(wejscia=3, ukryte=5, wyjscia=2, lr=0.6)
print("\nTrening dla nowego zbioru:")
siec2.trenuj(X_nowe, Y_nowe, max_epok=10000, docelowy_mse=0.001)

for x in X_nowe:
    _, wynik = siec2.w_przod(x)
    print(f"Wejście {x} -> {[round(w, 3) for w in wynik]}")