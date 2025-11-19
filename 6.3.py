import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# --- 1. Parametry Projektowe ---
F_s = 200     # Częstotliwość próbkowania (Hz)
F_0 = 50      # Częstotliwość do eliminacji (Hz)

# Obliczenie znormalizowanej częstotliwości kątowej omega
omega_norm = 2 * np.pi * (F_0 / F_s)
# omega_norm = np.pi / 2

# Wybór promieni
r1 = 1.0    # Promień zer: na okręgu jednostkowym dla eliminacji
r2 = 0.95   # Promień biegunów: blisko 1, ale wewnątrz dla stabilności

print(f"Projekt filtru Notch: F0={F_0} Hz, Fs={F_s} Hz")
print(f"Znormalizowana częstotliwość omega: {omega_norm:.4f} rad (pi/2)")
print(f"Parametry filtru: r1={r1}, r2={r2}")

# --- 2. Obliczenie współczynników filtru a i b ---

# Zera (Zeros)
# z1 = r1 * exp(j*omega_norm)
# z2 = r1 * exp(-j*omega_norm)
# Mianownik transmitancji B(z) = z^2 - (z1 + z2)*z + z1*z2
# Suma: 2 * r1 * cos(omega_norm)
# Iloczyn: r1^2

b = np.array([1, 
              -2 * r1 * np.cos(omega_norm), 
              r1**2])

# Bieguny (Poles)
# p1 = r2 * exp(j*omega_norm)
# p2 = r2 * exp(-j*omega_norm)
# Mianownik transmitancji A(z) = z^2 - (p1 + p2)*z + p1*p2
# Suma: 2 * r2 * cos(omega_norm)
# Iloczyn: r2^2

a = np.array([1, 
              -2 * r2 * np.cos(omega_norm), 
              r2**2])

print(f"\nWspółczynniki licznika (b): {b}")
print(f"Współczynniki mianownika (a): {a}")


# --- 3. Obserwacja odpowiedzi impulsowej (Impulse Response) ---

# Odpowiedź impulsowa to odpowiedź systemu na impuls jednostkowy delta[n]
impulse = np.zeros(100)
impulse[0] = 1.0
n_impulse = np.arange(100)

# Użycie funkcji lfilter do obliczenia odpowiedzi
h_n = signal.lfilter(b, a, impulse)

plt.figure(figsize=(12, 5))
plt.stem(n_impulse, h_n, linefmt='b-', markerfmt='bo', basefmt=" ")
plt.title('Odpowiedź impulsowa $h[n]$ filtru Notch (r2 < 1, stabilny IIR)')
plt.xlabel('n (numer próbki)')
plt.ylabel('Amplituda $h[n]$')
plt.grid(True)
plt.show()


# --- 4. Obserwacja charakterystyki amplitudowej (Magnitude Response) ---

# Obliczenie charakterystyki częstotliwościowej
# Parametr 'whole=True' to całe koło jednostkowe (0 do 2pi)
w, H = signal.freqz(b, a, worN=8192, whole=True) 

# Przekształcenie częstotliwości kątowej (w) na herce (F)
F_hz = w * F_s / (2 * np.pi)

# Amplituda (moduł) charakterystyki
magnitude_H = np.abs(H)

# Charakterystyka w decybelach (dB)
magnitude_dB = 20 * np.log10(magnitude_H / np.max(magnitude_H))


plt.figure(figsize=(12, 5))
plt.plot(F_hz, magnitude_dB)
# Zaznaczenie miejsca tłumienia (50 Hz)
plt.axvline(F_0, color='r', linestyle='--', label=f'Częstotliwość Notch: {F_0} Hz')
plt.axvline(F_s - F_0, color='r', linestyle='--') # Symetryczny do Fs/2
plt.axhline(-60, color='gray', linestyle=':') # Poziom tłumienia

plt.title('Charakterystyka Amplitudowa Filtru Notch (w dB)')
plt.xlabel('Częstotliwość (Hz)')
plt.ylabel('Wzmocnienie (dB)')
plt.xlim(0, F_s / 2) # Ograniczenie do pasma Nyquista
plt.ylim(-65, 5) 
plt.legend()
plt.grid(True)
plt.show()

# --- Wniosek ---
# Charakterystyka amplitudowa wykazuje bardzo głębokie minimum (notch) 
# dokładnie przy 50 Hz, co potwierdza eliminację tej częstotliwości.