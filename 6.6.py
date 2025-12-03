import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.signal import freqz

# Ustawienia wykresów
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['axes.grid'] = True
plt.style.use('seaborn-v0_8-whitegrid')

# --- Zadanie 1: Projektowanie metodą okna (rectangular) dla różnych M ---
Fs = 4000  # Częstotliwość próbkowania [Hz]
Fc = 1000  # Częstotliwość odcięcia [Hz]
wc_norm = Fc / (Fs / 2)  # Znormalizowana częstość odcięcia (względem Nyquista)

M_list = [11, 51, 101]  # Długości okna (rząd filtru M = numtaps - 1)

plt.figure()
for M in M_list:
    # firwin dla okna prostokątnego (rectangular)
    # Długość filtru: M+1. numtaps = M. length = M+1.
    # W scipy.signal.firwin M jest rzędem filtru (numtaps-1) lub długością numtaps
    # Używamy numtaps = M, co daje M współczynników (rząd M-1). L=M.
    # Użycie M+1 jest bliższe konwencji fir1(M,...) i daje rząd M, co jest użyteczne.
    # Będziemy używać M jako RZĘDU (M+1 współczynników)
    numtaps = M + 1 # Liczba współczynników (taps)
    
    # Okno prostokątne w firwin to 'boxcar'
    h_rect = signal.firwin(numtaps, cutoff=wc_norm, window='boxcar', pass_zero='lowpass', fs=Fs)
    
    # Obliczenie charakterystyki częstotliwościowej
    w_rect, H_rect = signal.freqz(h_rect, worN=2000, fs=Fs)
    
    # Wykres charakterystyki amplitudowej w dB
    plt.plot(w_rect, 20 * np.log10(np.abs(H_rect)), label=f'Prostokątne, Rząd M={M}')

plt.title('Charakterystyka Amplitudowa: Filtr Dolnoprzepustowy (Okna Prostokątne)')
plt.xlabel('Częstotliwość [Hz]')
plt.ylabel('Amplituda [dB]')
plt.ylim(-80, 5)
plt.axvline(Fc, color='red', linestyle='--', linewidth=0.8, label=f'Fc={Fc} Hz')
plt.legend()
plt.show()

# --- Zadanie 2: Porównanie różnych okien dla tej samej długości M ---
M_compare = 50  # Rząd filtru (51 współczynników)
numtaps_compare = M_compare + 1

plt.figure()

# Okno Prostokątne
h_rect_c = signal.firwin(numtaps_compare, cutoff=wc_norm, window='boxcar', pass_zero='lowpass', fs=Fs)
w_rect_c, H_rect_c = signal.freqz(h_rect_c, worN=2000, fs=Fs)
plt.plot(w_rect_c, 20 * np.log10(np.abs(H_rect_c)), label=f'Prostokątne, M={M_compare}')

# Okno Hamminga
h_hamm = signal.firwin(numtaps_compare, cutoff=wc_norm, window='hamming', pass_zero='lowpass', fs=Fs)
w_hamm, H_hamm = signal.freqz(h_hamm, worN=2000, fs=Fs)
plt.plot(w_hamm, 20 * np.log10(np.abs(H_hamm)), label=f'Hamminga, M={M_compare}')

plt.title(f'Charakterystyka Amplitudowa: Porównanie Okien (Rząd M={M_compare})')
plt.xlabel('Częstotliwość [Hz]')
plt.ylabel('Amplituda [dB]')
plt.ylim(-80, 5)
plt.axvline(Fc, color='red', linestyle='--', linewidth=0.8, label=f'Fc={Fc} Hz')
plt.legend()
plt.show()

# --- Zadanie 3: Filtr z zadanymi parametrami (Kaiser i Parks-McClellan) ---
Fs_spec = 5000  # Częstotliwość próbkowania [Hz]
Fpass = 1000  # Częstotliwość narożna pasma przepustowego [Hz]
Fstop = 1500  # Częstotliwość narożna pasma zaporowego [Hz]
Rp = 1  # Tętnienia w pasmie przepustowym [dB]
Rs = 30  # Tłumienie w pasmie zaporowym [dB]

# Znormalizowane częstotliwości
nyquist = Fs_spec / 2
Wpass = Fpass / nyquist
Wstop = Fstop / nyquist

# --- A) Projektowanie metodą okna Kaisera (Kaiserord i firwin) ---

# Tłumienie (attenuation) w dB to maksymalne tętnienia w pasmie przepustowym Lmax = Rp oraz 
# minimalne tłumienie w pasmie zaporowym Lmin = Rs.
# W kaiserord (Matlab) Rs to minimalne tłumienie w pasmie zaporowym w dB.
# Delta_p i Delta_s (max. błąd w pasmach)
delta_p = (10**(Rp/20) - 1) / (10**(Rp/20) + 1)
delta_s = 10**(-Rs/20)
A = -20 * np.log10(min(delta_p, delta_s)) # Rs jako minimalne tłumienie

# Oszacowanie rzędu filtru i parametru beta
# kaiserord przyjmuje: Wpass (passband freq), Wstop (stopband freq), Rs (stopband attenuation)
# W SciPy używamy oszacowania 'rem_att' jako minimalnego tłumienia w stopband.
# delta_f to szerokość pasma przejściowego (Fstop - Fpass)
numtaps_kaiser, beta_kaiser = signal.kaiserord(Rs, (Fstop - Fpass) / nyquist)
M_kaiser = numtaps_kaiser - 1

# Projektowanie filtru
h_kaiser = signal.firwin(numtaps_kaiser, cutoff=Fpass, window=('kaiser', beta_kaiser), pass_zero='lowpass', fs=Fs_spec)

print(f"\n--- Filtr Kaisera ---")
print(f"Minimalny Rząd (M): {M_kaiser}")
print(f"Liczba współczynników (numtaps): {numtaps_kaiser}")
print(f"Parametr beta: {beta_kaiser:.4f}")

# --- B) Projektowanie algorytmem Parks-McClellan (remez) ---

# Użycie funkcji remez (ekwiwalent firpm w Matlabie)
# bands: wektor częstotliwości granicznych (0, Fpass, Fstop, Nyquist)
bands = [0, Fpass, Fstop, nyquist]

# desired: pożądana amplituda w pasmach (1 w passband, 0 w stopband)
desired = [1, 0] 

# Wagi (weights): 
# Obliczone na podstawie pożądanych tętnień/tłumienia. Wagi są odwrotnie proporcjonalne do maksymalnego dopuszczalnego błędu.
# Max błąd w passband (delta_p): (10^(Rp/20) - 1) / 20log10(e) -> nie
# Max błąd w passband (liniowy): delta_p
# Max błąd w stopband (liniowy): delta_s
# Stosunek wag: Ws/Wp = delta_p / delta_s
weights = [1/delta_p, 1/delta_s]
W = [delta_s / delta_p, 1.0] # Wagi dla [Passband, Stopband] - remez wymaga wag Wp, Ws

# Oszacowanie rzędu (M) za pomocą funkcji remezord (nie istnieje w SciPy, ale można użyć oszacowania z kaiserord)
# Często dla remez jest to nieco niższe, ale możemy użyć jako dobry punkt startowy.
# Lepszym sposobem jest oszacowanie ze wzoru:
# M ≈ (-20*log10(sqrt(delta_p*delta_s)) - 13) / (2.32 * delta_f_norm)
# gdzie delta_f_norm = (Fstop - Fpass) / Fs

# W SciPy nie ma funkcji do automatycznego obliczania minimalnego rzędu dla remez/firpm.
# Użyjemy Rzędu z kaiserord i sprawdzimy, czy działa.
# numtaps_remez = numtaps_kaiser # Wymagane jest, aby remez zwrócił tę samą (lub niższą) wartość błędu.
# Dla remez w SciPy, numtaps to RZĄD filtru (M), więc numtaps_remez jest rzędem M
# W remez numtaps to **liczba współczynników (taps)**, co odpowiada M+1
numtaps_remez = numtaps_kaiser 
M_remez = numtaps_remez - 1

h_remez = signal.remez(numtaps_remez, bands, desired, weight=W, fs=Fs_spec, pass_zero='lowpass')

print(f"\n--- Filtr Parks-McClellan (remez) ---")
print(f"Użyty Rząd (M): {M_remez}")
print(f"Liczba współczynników (numtaps): {numtaps_remez}")
# Oszacowanie rzędu jest zwykle iteracyjne lub heurystyczne. Użycie kaiserord jest typowe.

# --- C) Wykres i weryfikacja ---

plt.figure()

# Filtr Kaisera
w_kaiser, H_kaiser = signal.freqz(h_kaiser, worN=2000, fs=Fs_spec)
H_kaiser_dB = 20 * np.log10(np.abs(H_kaiser))
plt.plot(w_kaiser, H_kaiser_dB, label=f'Kaiser, Rząd M={M_kaiser}')

# Filtr Parks-McClellan
w_remez, H_remez = signal.freqz(h_remez, worN=2000, fs=Fs_spec)
H_remez_dB = 20 * np.log10(np.abs(H_remez))
plt.plot(w_remez, H_remez_dB, label=f'Parks-McClellan, Rząd M={M_remez}')


# Linie referencyjne dla weryfikacji
# Passband: [0, Fpass], Max = Rp dB (1 dB), Min = -Rp dB (-1 dB)
# Stopband: [Fstop, Nyquist], Max = -Rs dB (-30 dB)
# Należy pamiętać, że remez projektuje z tolerancją błędu (ripple) równą e/W, 
# gdzie e jest minimalnym maksymalnym błędem ważonym.
plt.axvline(Fpass, color='green', linestyle=':', linewidth=1.5, label=f'Fpass={Fpass} Hz')
plt.axvline(Fstop, color='orange', linestyle=':', linewidth=1.5, label=f'Fstop={Fstop} Hz')

# Passband ripple (0 dB +/- Rp/2, jeśli projekt na 0 dB)
plt.axhline(Rp, color='green', linestyle='--', linewidth=0.8, alpha=0.7, label=f'Max Passband Ripple ({Rp} dB)')
plt.axhline(-Rp, color='green', linestyle='--', linewidth=0.8, alpha=0.7)

# Stopband attenuation (Max: -Rs dB)
plt.axhline(-Rs, color='orange', linestyle='--', linewidth=0.8, alpha=0.7, label=f'Min Stopband Attenuation ({-Rs} dB)')


plt.title('Charakterystyka Amplitudowa: Filtr Dolnoprzepustowy (Kaiser vs Parks-McClellan)')
plt.xlabel('Częstotliwość [Hz]')
plt.ylabel('Amplituda [dB]')
plt.ylim(-80, 5)
plt.legend()
plt.show()

# --- Weryfikacja spełnienia wymagań ---
# Sprawdzenie, czy faktyczne maks. tętnienia w pasmach mieszczą się w specyfikacji.

# Wartości w pasmie przepustowym (0 do Fpass)
passband_mask = (w_kaiser <= Fpass)
max_ripple_kaiser_pass = np.max(np.abs(20 * np.log10(np.abs(H_kaiser[passband_mask]))))
# Wartości w pasmie zaporowym (Fstop do Nyquist)
stopband_mask = (w_kaiser >= Fstop)
max_att_kaiser_stop = -np.max(20 * np.log10(np.abs(H_kaiser[stopband_mask])))

print(f"\n--- Weryfikacja Kaisera (Rząd M={M_kaiser}) ---")
print(f"Maks. tętnienia w Passband: {max_ripple_kaiser_pass:.2f} dB (Wymagane: < {Rp} dB) -> Spełnione: {max_ripple_kaiser_pass < Rp}")
print(f"Min. tłumienie w Stopband: {max_att_kaiser_stop:.2f} dB (Wymagane: > {Rs} dB) -> Spełnione: {max_att_kaiser_stop > Rs}")

passband_mask_remez = (w_remez <= Fpass)
max_ripple_remez_pass = np.max(np.abs(20 * np.log10(np.abs(H_remez[passband_mask_remez]))))
stopband_mask_remez = (w_remez >= Fstop)
max_att_remez_stop = -np.max(20 * np.log10(np.abs(H_remez[stopband_mask_remez])))

print(f"\n--- Weryfikacja Parks-McClellan (Rząd M={M_remez}) ---")
print(f"Maks. tętnienia w Passband: {max_ripple_remez_pass:.2f} dB (Wymagane: < {Rp} dB) -> Spełnione: {max_ripple_remez_pass < Rp}")
print(f"Min. tłumienie w Stopband: {max_att_remez_stop:.2f} dB (Wymagane: > {Rs} dB) -> Spełnione: {max_att_remez_stop > Rs}")