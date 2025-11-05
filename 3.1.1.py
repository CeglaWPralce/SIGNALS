import numpy as np
import pandas as pd

# --- Parametry ---
T0 = 1.0
T1 = 0.25
Nmax = 100
num_points = 5000
pi = np.pi


t = np.linspace(0, T0, num_points, endpoint=False)


def rectangular_signal(t, T0, T1):
    t_shifted = t - T0/2
    return np.where(np.abs(t_shifted) < T1, 1.0, 0.0)


def fourier_coeffs(T0, T1, N):
    k_vals = np.arange(-N, N+1)
    a_k = np.zeros_like(k_vals, dtype=float)
    for idx, k in enumerate(k_vals):
        if k == 0:
            a_k[idx] = 2 * T1 / T0
        else:
            a_k[idx] = np.sin(k * pi * T1 / T0) / (k * pi)
    return k_vals, a_k


def reconstruct(a_k, k_vals, t, T0):
    xN = np.zeros_like(t, dtype=complex)
    for k, ak in zip(k_vals, a_k):
        xN += ak * np.exp(1j * 2 * pi * k * (t - T0/2) / T0)
    return np.real(xN)


x_true = rectangular_signal(t, T0, T1)
errors = []

for N in range(Nmax + 1):
    k_vals, a_k = fourier_coeffs(T0, T1, N)
    xN = reconstruct(a_k, k_vals, t, T0)
    E_N = np.trapezoid(np.abs(x_true - xN)**2, t)
    errors.append(E_N)

df = pd.DataFrame({
    'N': np.arange(0, Nmax + 1),
    f'E_N (T1={T1})': errors
})

print(df.to_string(index=False, float_format='%.6f'))