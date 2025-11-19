import numpy as np
import matplotlib.pyplot as plt

# --- Parametry sygnału ---
A = 1.0          # Amplituda początkowa
phi = 0          # Faza (radiany)
omega_0 = np.pi / 10 # Częstotliwość kątowa (rad/próbka)
d = 0.05         # Współczynnik tłumienia (d > 0)
N = 100          # Liczba próbek
n = np.arange(N)

# --- 1. Sygnał z definicji (Równanie 5.5) ---
x_n_def = A * np.cos(omega_0 * n + phi) * np.exp(-d * n)

# --- 2. Sygnał z równania różnicowego (Równanie 5.4) ---

# Równanie: x[n] = 2*cos(omega_0)*exp(-d)*x[n-1] - exp(-2*d)*x[n-2]

# Obliczenie stałych współczynników
coeff_1 = 2 * np.cos(omega_0) * np.exp(-d)
coeff_2 = -np.exp(-2 * d) # Drugi człon to -exp(-2d) * x[n-2]

# Inicjalizacja tablicy na sygnał
x_n_diff = np.zeros(N)

# Warunki początkowe
# x[0] = A * cos(phi) * exp(-d*0) = A * cos(phi)
x_n_diff[0] = A * np.cos(phi)

# x[1] = A * cos(omega_0*1 + phi) * exp(-d*1)
x_n_diff[1] = A * np.cos(omega_0 * 1 + phi) * np.exp(-d * 1)


# Iteracyjne obliczenie kolejnych próbek z równania różnicowego
for k in range(2, N):
    x_n_diff[k] = coeff_1 * x_n_diff[k-1] + coeff_2 * x_n_diff[k-2]


# --- 3. Generowanie Wykresu Porównawczego ---

plt.figure(figsize=(12, 6))

# Wykres sygnału z definicji (jako linia ciągła - odniesienie)
plt.plot(n, x_n_def, 'r--', linewidth=2, label='Definicja (Równanie 5.5)')

# Wykres sygnału z równania różnicowego (jako dyskretne próbki)
plt.stem(n, x_n_diff, linefmt='b-', markerfmt='bo', basefmt=" ", label='Równanie różnicowe (Równanie 5.4)')

plt.title(f'5.2 Tłumiony Sygnał Sinusoidalny: Porównanie Metod 📉')
plt.xlabel('n (numer próbki)')
plt.ylabel('Amplituda $x[n]$')
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()

# --- Wniosek ---
# Dwa sygnały są nałożone idealnie, co udowadnia, że równanie różnicowe
# jest poprawnym modelem dla tłumionego sygnału sinusoidalnego.