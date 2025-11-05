import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Parametry filtra
f0 = 50.0       # częstotliwość tłumienia [Hz]
fs = 1000.0    # częstotliwość próbkowania [Hz] – wyższa dla gęstszego wykresu
Q = 30.0        # dobroć filtra

# Projekt filtra notch
b, a = signal.iirnotch(f0, Q, fs)

# Odpowiedź impulsowa – dłuższy czas i więcej punktów
N = int(0.5 * fs)   # 0.5 sekundy impulsu
impulse = np.zeros(N)
impulse[0] = 1
h = signal.lfilter(b, a, impulse)
t = np.arange(N) / fs

# Odpowiedź częstotliwościowa
f, H = signal.freqz(b, a, fs=fs, worN=80000)

# Rysowanie
plt.figure(figsize=(11,4))

plt.subplot(1,2,1)
plt.plot(t, h, 'b')
plt.title('Odpowiedź impulsowa h(t)')
plt.xlabel('t [s]')
plt.ylabel('h(t)')
plt.xlim(0, 0.5)
plt.grid(True, linestyle='--', alpha=0.5)

plt.subplot(1,2,2)
plt.plot(f, 20*np.log10(np.abs(H)), 'b')
plt.title('Charakterystyka amplitudowa |H(f)|')
plt.xlabel('F [Hz]')
plt.ylabel('|H(f)| [dB]')
plt.xlim(40, 60)
plt.ylim(-40, 5)
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()
