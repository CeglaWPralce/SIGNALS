import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# --- 1. Funkcja obliczająca charakterystykę częstotliwościową (Python analog MATLAB 'freqz_lab') ---
def freqz_lab(B, A, W):
    """
    Oblicza charakterystykę częstotliwościową H(exp(j*w)) dyskretnego systemu.
    
    H(z) = B(z) / A(z)
    H(exp(j*w)) = sum(B[m]*exp(-j*w*m)) / sum(A[k]*exp(-j*w*k))
    
    Argumenty:
      B (np.array): Współczynniki licznika (b[m]).
      A (np.array): Współczynniki mianownika (a[k]).
      W (np.array): Tablica znormalizowanych częstotliwości kątowych w radianach (omega).
      
    Zwraca:
      np.array: Tablica zespolonych wartości H(exp(j*w)).
    """
    
    H = np.zeros_like(W, dtype=complex)
    
    # Przechodzimy przez każdą częstotliwość w tablicy W
    for idx, w in enumerate(W):
        
        # Obliczenie E = exp(-j*w)
        E = np.exp(-1j * w)
        
        # --- Sumowanie Licznika B(E) ---
        B_val = 0.0 + 0.0j
        for m, b_coeff in enumerate(B):
            # Suma b[m] * E^(-m)
            B_val += b_coeff * (E ** m)
            
        # --- Sumowanie Mianownika A(E) ---
        A_val = 0.0 + 0.0j
        for k, a_coeff in enumerate(A):
            # Suma a[k] * E^(-k)
            A_val += a_coeff * (E ** k)
            
        # Ostateczna transmitancja
        if A_val != 0:
            H[idx] = B_val / A_val
        else:
            H[idx] = np.inf # Uniknięcie dzielenia przez zero (biegun na okręgu)
            
    return H

# --- 2. Przykładowe użycie: Filtr Uśredniający (FIR) ---

# Współczynniki dla filtru FIR: y[n] = 0.5*x[n] + 0.5*x[n-1]
B_coeffs_fir = np.array([0.5, 0.5]) # Współczynniki b[m]
A_coeffs_fir = np.array([1.0])      # Współczynniki a[k] (typowo a[0]=1 dla FIR)

# Generowanie wektora częstotliwości (od 0 do pi rad)
num_points = 512
W_rad = np.linspace(0, np.pi, num_points)

# --- 3. Obliczenia i Porównanie ---

# a) Obliczenie za pomocą własnej funkcji
H_manual = freqz_lab(B_coeffs_fir, A_coeffs_fir, W_rad)

# b) Obliczenie za pomocą wbudowanej funkcji SciPy (odpowiednik MATLAB 'freqz')
# Uwaga: signal.freqz zwraca również wektor częstotliwości w, ale użwamy W_rad dla zgodności
w_scipy, H_scipy = signal.freqz(B_coeffs_fir, A_coeffs_fir, worN=W_rad)


# --- 4. Wizualizacja Wyników ---

# Obliczenie Amplitudy (Modułu) i Fazy
Mag_manual = np.abs(H_manual)
Phase_manual = np.unwrap(np.angle(H_manual)) # unwrap do poprawnego wyświetlania fazy
Mag_scipy = np.abs(H_scipy)
Phase_scipy = np.unwrap(np.angle(H_scipy))

plt.figure(figsize=(12, 8))

# --- Charakterystyka Amplitudowa ---
plt.subplot(2, 1, 1)
plt.plot(W_rad, Mag_manual, 'b-', label='Własna funkcja (freqz_lab)')
plt.plot(W_rad, Mag_scipy, 'r--', alpha=0.7, label='scipy.signal.freqz')
plt.title('Charakterystyka Amplitudowa $|H(e^{j\omega})|$')
plt.xlabel('Częstotliwość Kątowa $\omega$ (rad/próbkę)')
plt.ylabel('Amplituda')
plt.legend()
plt.grid(True)

# --- Charakterystyka Fazowa ---
plt.subplot(2, 1, 2)
plt.plot(W_rad, Phase_manual, 'b-', label='Własna funkcja (freqz_lab)')
plt.plot(W_rad, Phase_scipy, 'r--', alpha=0.7, label='scipy.signal.freqz')
plt.title('Charakterystyka Fazowa $\measuredangle H(e^{j\omega})$')
plt.xlabel('Częstotliwość Kątowa $\omega$ (rad/próbkę)')
plt.ylabel('Faza (radiany)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Weryfikacja
print("-" * 40)
print("Czy moduły są identyczne (np.allclose)?", np.allclose(Mag_manual, Mag_scipy))
print("Czy fazy są identyczne (np.allclose)?", np.allclose(Phase_manual, Phase_scipy))