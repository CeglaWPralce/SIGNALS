import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# --- Parametry sygnału (wspólne dla wszystkich podpunktów) ---
Fs = 100        # Częstotliwość próbkowania w Hz
Ts = 1 / Fs     # Okres próbkowania w sekundach
F = 20          # Częstotliwość sygnału w Hz
DC = 1          # Składowa stała
A = 1           # Amplituda
phi = np.pi / 5 # Faza początkowa w radianach

# --- Funkcja pomocnicza do generowania sygnału ---
def generate_signal(N_samples):
    # n = 0, 1, ..., N_samples - 1
    n = np.arange(N_samples)
    t = n * Ts
    # x[n] = DC + A * cos(2*pi*F*t + phi)
    x = DC + A * np.cos(2 * np.pi * F * t + phi)
    return x, t

# --- Funkcja pomocnicza do tworzenia wykresu widma amplitudy ---
def plot_magnitude_spectrum(X, N_fft, Fs, title, ax):
    # Obliczanie osi częstotliwości w Hz
    f = fftfreq(N_fft, d=Ts)
    # Obliczanie widma amplitudy (normalizacja przez N_fft)
    # Wartość DC (f=0) jest X[0]
    # Amplituda A jest (2 * |X[k]|) / N_fft dla f > 0
    magnitude = np.abs(X)
    
    # Przesunięcie widma tak, aby 0 Hz było na środku
    magnitude_shifted = np.fft.fftshift(magnitude)
    f_shifted = np.fft.fftshift(f)
    
    # Skalowanie osi amplitudy (dla sygnału rzeczywistego)
    # Skalowanie DC
    DC_value = magnitude_shifted[f_shifted == 0] / N_fft
    # Skalowanie Amplitudy (wartości niezerowe, tylko dla dodatnich częstotliwości)
    # Szukamy tylko niezerowych wartości w zakresie f > 0
    # Zakładamy, że szczyty są symetryczne, więc bierzemy podwójną wartość
    # szczytu w widmie (właściwa amplituda to 2*|X[k]|/N_fft)
    scaled_magnitude = magnitude_shifted / N_fft
    scaled_magnitude[f_shifted != 0] = 2 * scaled_magnitude[f_shifted != 0]
    
    # Wykres tylko dla dodatnich częstotliwości (jednostronne widmo dla sygnałów rzeczywistych)
    positive_f = f_shifted > 0
    f_plot = f_shifted[positive_f]
    mag_plot = scaled_magnitude[positive_f]

    # Używamy stem do wizualizacji dyskretnych linii
    ax.stem(f_plot, mag_plot, basefmt=" ", linefmt='b-', markerfmt='bo')
    
    ax.set_title(title)
    ax.set_xlabel("Częstotliwość [Hz]")
    ax.set_ylabel("Amplituda w skali DC i A")
    ax.grid(True)
    
    # Ograniczenie osi dla lepszej widoczności
    ax.set_xlim(0, Fs/2) 
    ax.set_ylim(0, 1.2 * max(A, DC)) # Max na 1.2

# --- Funkcja do analizy i rysowania wyników ---
def analyze_and_plot(N_input, N_fft, window=None, subplot_index=1):
    x_input, t_input = generate_signal(N_input)
    
    # Wypełnianie zerami (Zero-padding)
    x_padded = np.zeros(N_fft)
    x_padded[:N_input] = x_input
    
    # Nakładanie okna, jeśli jest określone
    if window == 'rectangular':
        # Okno prostokątne to domyślne zachowanie (mnożenie przez jedynki)
        x_windowed = x_padded
        window_name = "prostokątne"
    elif window == 'hamming':
        # Okno Hamminga jest stosowane tylko do N_input próbek
        hamming_win = np.hamming(N_input)
        x_windowed = np.zeros(N_fft)
        x_windowed[:N_input] = x_input * hamming_win
        window_name = "Hamminga"
    else: # Dla podpunktów 1 i 2, traktujemy jako prostokątne na N_input
        x_windowed = x_padded
        window_name = "prostokątne"

    # Obliczenie DFT (za pomocą algorytmu FFT)
    X = fft(x_windowed, N_fft)
    
    # Tworzenie wykresu
    ax = fig.add_subplot(3, 2, subplot_index)
    
    if subplot_index in [1, 2]:
        title = f"N={N_input}, DFT: {N_fft}, Okno: {window_name}"
    elif subplot_index in [3, 4]:
        title = f"N={N_input}, DFT: {N_fft}, Okno: {window_name}"
    elif subplot_index in [5, 6]:
        title = f"N={N_input}, DFT: {N_fft}, Okno: {window_name}"

    plot_magnitude_spectrum(X, N_fft, Fs, title, ax)
    
    # Odczytywanie wartości DC i A (wartości oczekiwane: DC=1, A=1)
    
    # Składowa DC jest na f=0 Hz
    DC_idx = 0
    DC_read = np.abs(X[DC_idx]) / N_fft
    
    # Składowa A (częstotliwość F=20 Hz)
    # Indeks odpowiadający F=20 Hz to k = F * N_fft / Fs
    A_idx = int(F * N_fft / Fs)
    # Dla sygnału rzeczywistego, Amplituda = 2 * |X[k]| / N_fft
    A_read = 2 * np.abs(X[A_idx]) / N_fft
    
    return DC_read, A_read

# --- Inicjalizacja wykresu ---
fig = plt.figure(figsize=(14, 18))
plt.suptitle("Analiza DFT sygnału $x[n] = DC + A\cos(2\pi F t + \phi)$", fontsize=16)

# =================================================================
## 1. Test sygnału (7.5) z N=32 i N=64 (bez zero-paddingu)
# =================================================================
print("--- 1. N=32 (DFT z N=32, bez zero-paddingu) ---")
DC_32, A_32 = analyze_and_plot(N_input=32, N_fft=32, window='rectangular', subplot_index=1)
print(f"DC odczytane: {DC_32:.4f}, A odczytane: {A_32:.4f}")

print("\n--- 2. N=64 (DFT z N=64, bez zero-paddingu) ---")
DC_64, A_64 = analyze_and_plot(N_input=64, N_fft=64, window='rectangular', subplot_index=2)
print(f"DC odczytane: {DC_64:.4f}, A odczytane: {A_64:.4f}")

# =================================================================
## 2. Test sygnału (7.5) z N=64, zero-padding do 1024
# =================================================================
N_input_zp = 64
N_fft_zp = 1024

print("\n--- 3. N=64, DFT=1024 (Zero-padding, Okno prostokątne) ---")
DC_zp_rect, A_zp_rect = analyze_and_plot(N_input=N_input_zp, N_fft=N_fft_zp, window='rectangular', subplot_index=3)
print(f"DC odczytane: {DC_zp_rect:.4f}, A odczytane: {A_zp_rect:.4f}")

# =================================================================
## 3. Test sygnału (7.5) z N=64, zero-padding do 1024, Okna: Prostokątne vs Hamminga
# (Okno prostokątne jest już powyżej, rysujemy tylko Hamminga)
# =================================================================
print("\n--- 4. N=64, DFT=1024 (Zero-padding, Okno Hamminga) ---")
DC_zp_hamm, A_zp_hamm = analyze_and_plot(N_input=N_input_zp, N_fft=N_fft_zp, window='hamming', subplot_index=4)
print(f"DC odczytane: {DC_zp_hamm:.4f}, A odczytane: {A_zp_hamm:.4f}")

# =================================================================
## Sekcja dla sygnału (7.6) - Ograniczona do generowania i rozpoczęcia analizy
# Ze względu na złożoność i wielość podpunktów dla (7.6), 
# poniżej tylko przykładowa struktura
# =================================================================

print("\n\n--- Analiza sygnału złożonego (7.6) ---")
# x[n] = A_1 * cos(2*pi*F_1*t + phi_1) + A_2 * cos(2*pi*F_2*t + phi_2)
A1, F1, phi1 = 1, 20, 0
A2, F2, phi2 = 1, 40, 0 # Przykładowe wartości F2 dla pierwszego podpunktu
N_7_6_input = 256
N_7_6_fft = 1024
Fs_7_6 = 100 # Fs z zadania 7.5

# Generowanie sygnału 7.6
n_7_6 = np.arange(N_7_6_input)
t_7_6 = n_7_6 * Ts
x_7_6 = A1 * np.cos(2 * np.pi * F1 * t_7_6 + phi1) + \
        A2 * np.cos(2 * np.pi * F2 * t_7_6 + phi2)

# Wypełnianie zerami
x_7_6_padded = np.zeros(N_7_6_fft)
x_7_6_padded[:N_7_6_input] = x_7_6
X_7_6 = fft(x_7_6_padded, N_7_6_fft)

# Tworzenie wykresu dla sygnału 7.6 (A1=A2=1, F1=20, F2=40)
ax_7_6_a = fig.add_subplot(3, 2, 5)
plot_magnitude_spectrum(X_7_6, N_7_6_fft, Fs_7_6, 
                        f"Sygnał (7.6): N={N_7_6_input}, DFT={N_7_6_fft}, F1={F1}Hz, F2={F2}Hz",
                        ax_7_6_a)
print(f"Wyświetlono widmo dla A1=A2=1, F1=20Hz, F2=40Hz (Zero-padding do 1024)")


# Przykładowa analiza F1=1, F2=24Hz (N=256, DFT=1024) - Czysty wyciek widma
A2_new, F2_new = 1, 24
x_7_6_new = A1 * np.cos(2 * np.pi * F1 * t_7_6 + phi1) + \
            A2_new * np.cos(2 * np.pi * F2_new * t_7_6 + phi2)
x_7_6_padded_new = np.zeros(N_7_6_fft)
x_7_6_padded_new[:N_7_6_input] = x_7_6_new
X_7_6_new = fft(x_7_6_padded_new, N_7_6_fft)

ax_7_6_b = fig.add_subplot(3, 2, 6)
plot_magnitude_spectrum(X_7_6_new, N_7_6_fft, Fs_7_6,
                        f"Sygnał (7.6): N={N_7_6_input}, DFT={N_7_6_fft}, F1={F1}Hz, F2={F2_new}Hz",
                        ax_7_6_b)
print(f"Wyświetlono widmo dla A1=A2=1, F1=20Hz, F2=24Hz (Zero-padding do 1024, Wyciek widma)")


plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.show()