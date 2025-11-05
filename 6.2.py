import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter

# --- 1. Definicja funkcji LCCDE (Analog MATLAB 'filter') ---
def my_lfilter(b, a, x):
    """
    Implementuje liniowe równanie różnicowe o stałych współczynnikach (LCCDE).
    Równanie: a[0]*y[n] = sum(b[m]*x[n-m]) - sum(a[k]*y[n-k])
    
    Argumenty:
      b (np.array): Współczynniki wejściowe (licznika/FIR). b[0] do b[M].
      a (np.array): Współczynniki sprzężenia zwrotnego (mianownika/IIR). a[0] do a[N].
      x (np.array): Sygnał wejściowy x[n].
      
    Zwraca:
      np.array: Sygnał wyjściowy y[n].
    """
    N_x = len(x)
    M = len(b) - 1 # Rząd FIR
    N = len(a) - 1 # Rząd IIR
    
    # Inicjalizacja sygnału wyjściowego
    y = np.zeros(N_x)
    
    # Normalizacja współczynników b i a przez a[0]
    if a[0] == 0:
        raise ValueError("Współczynnik a[0] musi być niezerowy.")
        
    b_norm = b / a[0]
    a_norm = a / a[0] # a_norm[0] = 1

    # Główna pętla dla każdego n
    for n in range(N_x):
        
        # --- 1. Składnik Wejściowy (FIR) ---
        # Suma b[m]/a[0] * x[n-m]
        fir_term = 0.0
        for m in range(M + 1):
            x_index = n - m
            # x[n-m] jest brane z x tylko, gdy indeks >= 0
            if x_index >= 0:
                fir_term += b_norm[m] * x[x_index]
        
        # --- 2. Składnik Sprzężenia Zwrotnego (IIR) ---
        # Suma a[k]/a[0] * y[n-k] (k zaczyna się od 1)
        iir_term = 0.0
        for k in range(1, N + 1):
            y_index = n - k
            # y[n-k] jest brane z y tylko, gdy indeks >= 0
            if y_index >= 0:
                iir_term += a_norm[k] * y[y_index]
        
        # Ostateczne obliczenie y[n]
        y[n] = fir_term - iir_term
        
    return y

# --- 2. Przykładowe użycie: Filtr dolnoprzepustowy (IIR) ---

# Parametry filtru IIR (np. z filtru Butterworha drugiego rzędu)
# Zapewnia stabilny filtr z obu stron
b_coeff = np.array([0.0976, 0.1953, 0.0976])  # Współczynniki b (licznika)
a_coeff = np.array([1.0, -0.9428, 0.3333])    # Współczynniki a (mianownika), a[0]=1.0

# Sygnał wejściowy: sinusoida niskiej i wysokiej częstotliwości
N_x = 200
n_x = np.arange(N_x)
# Niska częstotliwość (przechodzi) + Wysoka częstotliwość (tłumiona)
x_in = np.sin(2 * np.pi * 0.05 * n_x) + np.sin(2 * np.pi * 0.4 * n_x) 


# --- 3. Porównanie wyników ---

# a) Obliczenie za pomocą własnej funkcji
y_manual = my_lfilter(b_coeff, a_coeff, x_in)

# b) Obliczenie za pomocą scipy.signal.lfilter (odpowiednik MATLAB 'filter')
y_scipy = lfilter(b_coeff, a_coeff, x_in)


# --- 4. Prezentacja i wizualizacja ---

print("Współczynniki b (licznika):", b_coeff)
print("Współczynniki a (mianownika):", a_coeff)
print("-" * 40)
print("Czy wyniki są numerycznie identyczne (np.allclose)?", np.allclose(y_manual, y_scipy))


plt.figure(figsize=(12, 8))

# Sygnał wejściowy
plt.subplot(3, 1, 1)
plt.plot(n_x, x_in, 'b-')
plt.title('Sygnał Wejściowy $x[n]$ (Niska + Wysoka częstotliwość)')
plt.ylabel('Amplituda')
plt.grid(True)

# Wynik - Własna implementacja
plt.subplot(3, 1, 2)
plt.plot(n_x, y_manual, 'g-')
plt.title('Sygnał Wyjściowy $y[n]$ - Własna Implementacja LCCDE (Dolnoprzepustowy)')
plt.ylabel('Amplituda')
plt.grid(True)

# Porównanie
plt.subplot(3, 1, 3)
plt.plot(n_x, y_manual, 'g-', label='Własna implementacja')
plt.plot(n_x, y_scipy, 'r--', alpha=0.6, label='scipy.signal.lfilter')
plt.title('Porównanie Wyników')
plt.xlabel('n (indeks próbki)')
plt.ylabel('Amplituda')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()