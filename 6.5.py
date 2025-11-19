import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# --- 1. Parametry Projektowe (Znormalizowane) ---
Fs = 5000       # Częstotliwość próbkowania
Wp = 1000 / (Fs / 2) # 0.4
Ws = 1500 / (Fs / 2) # 0.6
Rp = 1          # Tętnienie pasma przepustowego (dB)
Rs = 30         # Tłumienie pasma zaporowego (dB)
Nyquist = Fs / 2

# Lista do przechowywania współczynników i rzędów
filters = {}

# --- 2. Projekt i Obliczenie Rzędów Filtrów ---
print("--- Rzędy i Transmitancje Filtrów IIR ---")

# 2.1. Butterworth
N_butter, Wn_butter = signal.buttord(Wp, Ws, Rp, Rs)
b_butter, a_butter = signal.butter(N_butter, Wn_butter, 'low', analog=False)
filters['Butterworth'] = {'N': N_butter, 'b': b_butter, 'a': a_butter}
print(f"Butterworth: Rząd N={N_butter}")

# 2.2. Chebyshev Typ I
N_cheby1, Wn_cheby1 = signal.cheb1ord(Wp, Ws, Rp, Rs)
b_cheby1, a_cheby1 = signal.cheby1(N_cheby1, Rp, Wp, 'low', analog=False)
filters['Chebyshev I'] = {'N': N_cheby1, 'b': b_cheby1, 'a': a_cheby1}
print(f"Chebyshev I: Rząd N={N_cheby1}")

# 2.3. Chebyshev Typ II
N_cheby2, Wn_cheby2 = signal.cheb2ord(Wp, Ws, Rp, Rs)
b_cheby2, a_cheby2 = signal.cheby2(N_cheby2, Rs, Ws, 'low', analog=False)
filters['Chebyshev II'] = {'N': N_cheby2, 'b': b_cheby2, 'a': a_cheby2}
print(f"Chebyshev II: Rząd N={N_cheby2}")

# 2.4. Elliptic (Cauer)
N_ellip, Wn_ellip = signal.ellipord(Wp, Ws, Rp, Rs)
b_ellip, a_ellip = signal.ellip(N_ellip, Rp, Rs, Wp, 'low', analog=False)
filters['Elliptic'] = {'N': N_ellip, 'b': b_ellip, 'a': a_ellip}
print(f"Elliptic: Rząd N={N_ellip}")

# --- 3. Wykres Zer i Biegunów ---
plt.figure(figsize=(10, 10))
plt.title('3. Zera i Bieguny Filtrów IIR (płaszczyzna Z) ')
plt.axhline(0, color='gray', linestyle='--')
plt.axvline(0, color='gray', linestyle='--')
plt.gca().add_patch(plt.Circle((0, 0), 1, color='lightgray', fill=False, linestyle='--'))
plt.xlabel('Re(z)')
plt.ylabel('Im(z)')
plt.axis('equal')
plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)

markers = ['o', 'x', 's', 'd']
colors = ['blue', 'red', 'green', 'purple']
legend_entries = []

for i, (name, data) in enumerate(filters.items()):
    b, a = data['b'], data['a']
    # Obliczenie zer (z) i biegunów (p)
    z, p, k = signal.tf2zpk(b, a)
    
    # Rysowanie biegunów (x)
    plt.plot(np.real(p), np.imag(p), marker='x', color=colors[i], linestyle='None', markersize=8, markeredgewidth=1.5, label=f'{name} Bieguny')
    
    # Rysowanie zer (o)
    plt.plot(np.real(z), np.imag(z), marker='o', color=colors[i], linestyle='None', markersize=8, fillstyle='none', markeredgewidth=1.5, label=f'{name} Zera')

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# --- 4. Wykres Charakterystyk Amplitudowych i Fazowych ---
plt.figure(figsize=(12, 10))

# Wygenerowanie wektora częstotliwości do analizy (do Nyquista)
w_rad, F_amp = signal.freqz(filters['Butterworth']['b'], filters['Butterworth']['a'], worN=8000)
F_hz = w_rad * Nyquist / np.pi # Przekształcenie z rad na Hz

# Iteracja i rysowanie
for i, (name, data) in enumerate(filters.items()):
    b, a = data['b'], data['a']
    # Obliczenie odpowiedzi częstotliwościowej
    w, H = signal.freqz(b, a, worN=w_rad)
    Mag_dB = 20 * np.log10(np.abs(H))
    Phase = np.unwrap(np.angle(H)) * 180 / np.pi # Faza w stopniach
    
    # --- Charakterystyka Amplitudowa ---
    plt.subplot(2, 1, 1)
    plt.plot(F_hz, Mag_dB, label=f'{name} (N={data["N"]})', color=colors[i])
    
    # --- Charakterystyka Fazowa ---
    plt.subplot(2, 1, 2)
    plt.plot(F_hz, Phase, label=f'{name} (N={data["N"]})', color=colors[i])

# Ustawienia wykresu Amplitudowego
plt.subplot(2, 1, 1)
plt.title('4. Porównanie Charakterystyk Amplitudowych Filtrów IIR')
plt.xlabel('Częstotliwość (Hz)')
plt.ylabel('Wzmocnienie (dB)')
# Weryfikacja spełnienia wymagań
plt.axvline(Fs/2 * Wp, color='k', linestyle=':', label='Fpass=1kHz')
plt.axvline(Fs/2 * Ws, color='k', linestyle=':', label='Fstop=1.5kHz')
plt.axhline(-Rp, color='r', linestyle='--')
plt.axhline(-Rs, color='r', linestyle='--')
plt.ylim(-60, 5) 
plt.legend()
plt.grid(True)

# Ustawienia wykresu Fazowego
plt.subplot(2, 1, 2)
plt.title('5. Porównanie Charakterystyk Fazowych Filtrów IIR')
plt.xlabel('Częstotliwość (Hz)')
plt.ylabel('Faza (stopnie)')
plt.axvline(Fs/2 * Wp, color='k', linestyle=':')
plt.axvline(Fs/2 * Ws, color='k', linestyle=':')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# --- 5. Wykres Odpowiedzi Impulsowej ---
plt.figure(figsize=(12, 6))
N_impulse = 50 # Liczba próbek
impulse = np.zeros(N_impulse)
impulse[0] = 1.0
n_impulse = np.arange(N_impulse)

plt.title('6. Porównanie Odpowiedzi Impulsowej Filtrów IIR ')
plt.xlabel('n (numer próbki)')
plt.ylabel('Amplituda $h[n]$')

for i, (name, data) in enumerate(filters.items()):
    b, a = data['b'], data['a']
    # Użycie signal.lfilter (odpowiednik MATLAB 'filter')
    h_n = signal.lfilter(b, a, impulse) 
    
    # Użycie stem (dla sygnałów dyskretnych)
    plt.stem(n_impulse, h_n, linefmt=f'{colors[i]}-', markerfmt=f'{colors[i]}o', basefmt=" ", label=f'{name} (N={data["N"]})', alpha=0.7)

plt.legend()
plt.grid(True)
plt.show()