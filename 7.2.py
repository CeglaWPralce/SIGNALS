import numpy as np
import matplotlib.pyplot as plt

# --- 1. Definicja funkcji DFT i IDFT ---

def dft(x):
    """
    Oblicza Dyskretną Transformację Fouriera (DFT) sygnału x
    według wzoru (7.2).
    """
    N = len(x)
    n = np.arange(N)
    k = np.arange(N)
    # Macierz W_N = exp(-j * 2 * pi * k * n / N)
    W_N = np.exp(-2j * np.pi * np.outer(k, n) / N)
    # X[k] = suma_n=0^N-1 x[n] * W_N^(k*n)
    X = W_N @ x
    return X

def idft(X):
    """
    Oblicza Odwrotną Dyskretną Transformację Fouriera (IDFT) widma X
    według wzoru (7.3).
    """
    N = len(X)
    n = np.arange(N)
    k = np.arange(N)
    # Macierz W_N_inv = exp(j * 2 * pi * k * n / N)
    W_N_inv = np.exp(2j * np.pi * np.outer(n, k) / N)
    # x[n] = 1/N * suma_k=0^N-1 X[k] * W_N_inv^(n*k)
    x = (1/N) * (W_N_inv @ X)
    # Zwracamy część rzeczywistą, ponieważ sygnał wejściowy x był rzeczywisty
    return x.real

# --- 2. Sygnał testowy (sinusoidalny) ---

# Parametry sygnału
fs = 1000        # Częstotliwość próbkowania (Hz)
T = 1/fs         # Okres próbkowania
L = 128          # Długość sygnału (Liczba próbek), musi być potęgą 2 dla efektywności FFT, ale DFT działa dla dowolnej L
t = np.linspace(0, (L-1)*T, L, endpoint=False) # Wektor czasu
f1 = 50          # Częstotliwość sygnału 1 (Hz)
f2 = 120         # Częstotliwość sygnału 2 (Hz)

# Sygnał wejściowy (suma dwóch sinusoid)
x_n = 0.7 * np.sin(2 * np.pi * f1 * t) + 1.0 * np.sin(2 * np.pi * f2 * t)

# --- 3. Obliczenia i porównania ---

# a) DFT i IDFT zaimplementowane
X_k_dft = dft(x_n)
x_n_idft = idft(X_k_dft)

# b) DFT i IDFT za pomocą wbudowanej funkcji NumPy (FFT)
X_k_fft = np.fft.fft(x_n)
x_n_ifft = np.fft.ifft(X_k_fft).real

# c) Błąd rekonstrukcji (według wzoru 7.4)
# DFT i IDFT są swoimi wzajemnymi odwrotnościami,
# więc IDFT(DFT(x[n])) powinno być równe x[n].
# Błąd liczymy dla naszej implementacji IDFT(DFT(x[n]))
error_reconstruction = x_n - x_n_idft
epsilon = np.abs(error_reconstruction) # Wzór (7.4): epsilon = |x[n] - IDFT(DFT{x[n]})|

# d) Błąd porównawczy między naszą DFT a funkcją fft
error_dft_vs_fft = X_k_dft - X_k_fft
error_idft_vs_ifft = x_n_idft - x_n_ifft

# --- 4. Wyniki i Wykresy ---

print(f"Norma błędu rekonstrukcji (max|epsilon|): {np.max(epsilon):.10f}")
print(f"Norma błędu |DFT - FFT| (max): {np.max(np.abs(error_dft_vs_fft)):.10f}")
print("-" * 50)

# Użycie funkcji subplot do tworzenia wielu wykresów
fig, axes = plt.subplots(4, 1, figsize=(10, 12))
plt.suptitle('Analiza Dyskretnej Transformacji Fouriera (DFT)', fontsize=16)

# Wykres 1: Sygnał wejściowy i rekonstruowany
axes[0].plot(t * 1000, x_n, label='Sygnał wejściowy $x[n]$', color='blue')
axes[0].plot(t * 1000, x_n_idft, label='Sygnał zrekonstruowany IDFT(DFT($x[n]$))', color='red', linestyle='dashed', alpha=0.7)
axes[0].set_title('Sygnał wejściowy i rekonstruowany')
axes[0].set_xlabel('Czas [ms]')
axes[0].set_ylabel('Amplituda')
axes[0].legend()
axes[0].grid(True)

# Wykres 2: Widmo Amplitudowe (DFT vs FFT)
# Obliczamy wektor częstotliwości dla wykresu widma
f_axis = fs * np.arange(L) / L
axes[1].plot(f_axis[:L//2], np.abs(X_k_dft[:L//2]), label='DFT (nasza implementacja)', color='green', marker='o', linestyle='')
axes[1].plot(f_axis[:L//2], np.abs(X_k_fft[:L//2]), label='FFT (numpy.fft)', color='orange', linestyle='dashed')
axes[1].set_title('Widmo Amplitudowe: Nasze DFT vs. NumPy FFT')
axes[1].set_xlabel('Częstotliwość [Hz]')
axes[1].set_ylabel('Amplituda $|X[k]|$')
axes[1].legend()
axes[1].grid(True)

# Wykres 3: Błąd rekonstrukcji $\varepsilon = x[n] - \text{IDFT}(\text{DFT}\{x[n]\})$
# Wykres 3: Błąd rekonstrukcji $\varepsilon = x[n] - \text{IDFT}(\text{DFT}\{x[n]\})$
axes[2].stem(t * 1000, epsilon, 
             label='Błąd rekonstrukcji $\\epsilon$', 
             basefmt=" ", 
             linefmt='k-', 
             markerfmt='ko') # Argument 'use_line_collection' usunięty
axes[2].set_title('Błąd rekonstrukcji (Wzór 7.4)')
axes[2].set_xlabel('Czas [ms]')
axes[2].set_ylabel('$|\\epsilon|$')
axes[2].legend()
axes[2].grid(True)

# Wykres 4: Porównanie wyników IDFT (błąd między naszą a wbudowaną)
axes[3].plot(t * 1000, error_idft_vs_ifft, label='$x_{\\text{IDFT}} - x_{\\text{IFTT}}$', color='purple')
axes[3].set_title('Różnica między naszą IDFT a NumPy IFFT')
axes[3].set_xlabel('Czas [ms]')
axes[3].set_ylabel('Różnica')
axes[3].legend()
axes[3].grid(True)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()