import numpy as np
import matplotlib.pyplot as plt

# Wspólne parametry
A = 1.0  # Amplituda
phi = 0  # Faza (radiany)
N = 100  # Liczba próbek (n)
n = np.arange(N)

# --- 1. Sygnał sinusoidalny: x[n] = A*cos(omega_0*n + phi) ---

# Ustawienia dla pierwszego porównania
omega_0_a = np.pi / 4  # Częstotliwość kątowa 1 (rad/próbka)
omega_0_b = omega_0_a + 2 * np.pi  # Częstotliwość kątowa 2, różniąca się o 2pi

# Obliczenie sygnałów
x_n_a = A * np.cos(omega_0_a * n + phi)
x_n_b = A * np.cos(omega_0_b * n + phi)

plt.figure(figsize=(10, 6))
plt.stem(n, x_n_a, linefmt='b-', markerfmt='bo', basefmt=" ")
plt.stem(n, x_n_b, linefmt='r--', markerfmt='rx', basefmt=" ")
plt.title(f'1. Porównanie sygnałów: $x[n] = A\cos(\omega_0 n + \phi)$')
plt.xlabel('n (numer próbki)')
plt.ylabel('Amplituda')
plt.legend([f'$\omega_0 = {omega_0_a:.3f}$ rad', f'$\omega_0 + 2\pi = {omega_0_b:.3f}$ rad'])
plt.grid(True)
plt.tight_layout()
plt.show()

# Wniosek: Sygnały są identyczne, ponieważ e^j(omega_0 + 2pi)n = e^j(omega_0)n

# --- 2. Sygnał sinusoidalny: x[n] = A*cos(2*pi*(F_0/F_s)*n + phi) ---

# Ustawienia dla F_0 < F_s/2
F_s = 100  # Częstotliwość próbkowania (Hz)
F_0_1 = 5   # F_0 < F_s/2 = 50 Hz
F_0_2 = 40  # Inna F_0 < F_s/2

# Obliczenie sygnałów
x_n_f1 = A * np.cos(2 * np.pi * (F_0_1 / F_s) * n + phi)
x_n_f2 = A * np.cos(2 * np.pi * (F_0_2 / F_s) * n + phi)

plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.stem(n, x_n_f1, linefmt='b-', markerfmt='bo', basefmt=" ")
plt.title(f'2. Sygnał: $F_0 = {F_0_1}$ Hz ($\omega_0 = 2\pi \cdot {F_0_1/F_s:.2f}$ rad)')
plt.ylabel('Amplituda')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.stem(n, x_n_f2, linefmt='r-', markerfmt='rx', basefmt=" ")
plt.title(f'Sygnał: $F_0 = {F_0_2}$ Hz ($\omega_0 = 2\pi \cdot {F_0_2/F_s:.2f}$ rad)')
plt.xlabel('n (numer próbki)')
plt.ylabel('Amplituda')
plt.grid(True)

plt.tight_layout()
plt.show()

# --- 3. Sygnał sinusoidalny z równania różnicowego (Difference Equation) ---

# Ustawienia dla równania różnicowego
omega_0_diff = np.pi / 8
cos_omega_0 = np.cos(omega_0_diff)

# Warunki początkowe dla x[n] = A*cos(omega_0*n + phi):
# x[0] = A * cos(phi)
# x[1] = A * cos(omega_0 + phi)
x_n_diff = np.zeros(N)
x_n_diff[0] = A * np.cos(phi)
x_n_diff[1] = A * np.cos(omega_0_diff * 1 + phi)

# Równanie różnicowe: x[n] = 2*cos(omega_0)*x[n-1] - x[n-2]
for k in range(2, N):
    x_n_diff[k] = 2 * cos_omega_0 * x_n_diff[k-1] - x_n_diff[k-2]

# Weryfikacja (dla porównania obliczony jawny wzór)
x_n_explicit = A * np.cos(omega_0_diff * n + phi)

plt.figure(figsize=(10, 6))
plt.stem(n, x_n_diff, linefmt='g-', markerfmt='go', basefmt=" ")
plt.plot(n, x_n_explicit, 'r--', label='Jawny wzór (A*cos(omega_0*n + phi))')
plt.title(f'3. Sygnał z równania różnicowego: $\omega_0 = {omega_0_diff:.3f}$ rad')
plt.xlabel('n (numer próbki)')
plt.ylabel('Amplituda')
plt.legend(['Równanie różnicowe', 'Wzór jawny'])
plt.grid(True)
plt.tight_layout()
plt.show()

# Wniosek: Sygnał wygenerowany równaniem różnicowym jest zgodny z sygnałem sinusoidalnym (dokładność zależy od precyzji zmiennych).