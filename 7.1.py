import numpy as np
import matplotlib.pyplot as plt

# --- Ustawienia Globalne ---
M = 20  # Długość sygnału: od -M do M (łącznie 2*M + 1 próbek)
N_fft = 2048  # Liczba punktów do oceny DTFT (dla gładkiego wykresu)
w0_exp = np.pi / 4  # Częstotliwość dla sygnału wykładniczego
w0_cos = np.pi / 8  # Częstotliwość dla sygnału sinusoidalnego

# Wektor indeksów czasowych n
n = np.arange(-M, M + 1)

# Wektor częstotliwości omega (od -pi do pi)
w = np.linspace(-np.pi, np.pi, N_fft, endpoint=False) # Endpoint=False zapewnia niepowtarzanie pi

# Macierz e^(-j*w*n) (dla szybkiego obliczenia DTFT)
# Wymiar: N_fft x (2*M + 1)
# np.outer tworzy macierz iloczynu zewnętrznego: w[i] * n[j]
exponent = -1j * np.outer(w, n)
E = np.exp(exponent)

## 1. Sygnał stały: x_M[n] = 1, -M <= n <= M
# Wektor sygnału (wiersz)
x_const = np.ones_like(n, dtype=float)

# Obliczenie DTFT: X(e^jw) = E @ x_const.T (mnożenie macierzy E przez wektor x_const transponowany)
X_const = E @ x_const

## 2. Sygnał wykładniczy zespolony: x_M[n] = e^(j*w0*n), -M <= n <= M
x_exp = np.exp(1j * w0_exp * n)

# Obliczenie DTFT
X_exp = E @ x_exp

## 3. Sygnał sinusoidalny: x_M[n] = cos(w0*n), -M <= n <= M
x_cos = np.cos(w0_cos * n)

# Obliczenie DTFT
X_cos = E @ x_cos

# --- Rysowanie Charakterystyk Amplitudowych ---
fig, axes = plt.subplots(3, 1, figsize=(10, 10))
fig.suptitle(f'Charakterystyki Amplitudowe DTFT dla M = {M}', fontsize=16)

# 1. Sygnał Stały
axes[0].plot(w, np.abs(X_const))
axes[0].set_title('Sygnał Stały $x_M[n] = 1$')
axes[0].set_xlabel('$\omega$ (radiany/próbkę)')
axes[0].set_ylabel('$|X(e^{j\omega})|$')
axes[0].grid(True)

# 2. Sygnał Wykładniczy Zespolony
axes[1].plot(w, np.abs(X_exp))
axes[1].set_title(f'Sygnał Wykładniczy $x_M[n] = e^{{j\omega_0 n}}$ ($\omega_0 = \pi/4$)')
axes[1].axvline(w0_exp, color='r', linestyle='--', label='$\omega_0$')
axes[1].set_xlabel('$\omega$ (radiany/próbkę)')
axes[1].set_ylabel('$|X(e^{j\omega})|$')
axes[1].grid(True)
axes[1].legend()

# 3. Sygnał Sinusoidalny
axes[2].plot(w, np.abs(X_cos))
axes[2].set_title(f'Sygnał Sinusoidalny $x_M[n] = \cos(\omega_0 n)$ ($\omega_0 = \pi/8$)')
axes[2].axvline(w0_cos, color='r', linestyle='--', label='$\omega_0$')
axes[2].axvline(-w0_cos, color='r', linestyle='--')
axes[2].set_xlabel('$\omega$ (radiany/próbkę)')
axes[2].set_ylabel('$|X(e^{j\omega})|$')
axes[2].grid(True)
axes[2].legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Dostosowanie dla głównego tytułu
plt.show()