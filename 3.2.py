<<<<<<< HEAD
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift, fftfreq

# Parametry
Tmax = 10
Fs = 100
t = np.linspace(-Tmax, Tmax, int(2*Tmax*Fs))
dt = t[1] - t[0]

# 1) Prostokątny impuls
x1 = np.where(np.abs(t) < 1, 1, 0)

# 2) Fragment sinusoidy
om0 = 5 * np.pi
x2 = np.where(np.abs(t) < 1, np.cos(om0 * t), 0)

# 3) Fragment sygnału AM
kAM = 4
omm = np.pi
x3 = np.where(np.abs(t) < 1, (1 + kAM * np.cos(omm * t)) * np.cos(om0 * t), 0)

# 4) Sygnał zespolony
x4 = np.where(np.abs(t) < 1, np.exp(1j*omm*t), 0)

# Obliczanie FFT i osi częstotliwości
def fourier(x):
    X = fftshift(fft(x))
    f = fftshift(fftfreq(len(t), d=dt))
    return f, np.abs(X) / len(t)

signals = [x1, x2, x3, x4]
titles = [
    'Prostokątny impuls',
    'Fragment sinusoidy',
    'Sygnał z modulacją amplitudy (AM)',
    'Sygnał zespolony'
]

# Rysowanie
plt.figure(figsize=(12, 12))
for i, (sig, title) in enumerate(zip(signals, titles), start=1):
    f, X = fourier(sig)
    plt.subplot(4, 2, 2*i-1)
    plt.plot(t, sig.real)  # jeśli zespolony, bierzemy część rzeczywistą
    plt.title(f'{i}. {title} – sygnał czasowy')
    plt.xlabel('t [s]')
    plt.ylabel('x(t)')
    plt.grid(True)

    plt.subplot(4, 2, 2*i)
    plt.plot(f, X)
    plt.title(f'{i}. {title} – widmo amplitudowe')
    plt.xlabel('Częstotliwość [Hz]')
    plt.ylabel('|X(f)|')
    plt.grid(True)

plt.tight_layout()
=======
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift, fftfreq

# Parametry
Tmax = 10
Fs = 100
t = np.linspace(-Tmax, Tmax, int(2*Tmax*Fs))
dt = t[1] - t[0]

# 1) Prostokątny impuls
x1 = np.where(np.abs(t) < 1, 1, 0)

# 2) Fragment sinusoidy
om0 = 5 * np.pi
x2 = np.where(np.abs(t) < 1, np.cos(om0 * t), 0)

# 3) Fragment sygnału AM
kAM = 4
omm = np.pi
x3 = np.where(np.abs(t) < 1, (1 + kAM * np.cos(omm * t)) * np.cos(om0 * t), 0)

# 4) Sygnał zespolony
x4 = np.where(np.abs(t) < 1, np.exp(1j*omm*t), 0)

# Obliczanie FFT i osi częstotliwości
def fourier(x):
    X = fftshift(fft(x))
    f = fftshift(fftfreq(len(t), d=dt))
    return f, np.abs(X) / len(t)

signals = [x1, x2, x3, x4]
titles = [
    'Prostokątny impuls',
    'Fragment sinusoidy',
    'Sygnał z modulacją amplitudy (AM)',
    'Sygnał zespolony'
]

# Rysowanie
plt.figure(figsize=(12, 12))
for i, (sig, title) in enumerate(zip(signals, titles), start=1):
    f, X = fourier(sig)
    plt.subplot(4, 2, 2*i-1)
    plt.plot(t, sig.real)  # jeśli zespolony, bierzemy część rzeczywistą
    plt.title(f'{i}. {title} – sygnał czasowy')
    plt.xlabel('t [s]')
    plt.ylabel('x(t)')
    plt.grid(True)

    plt.subplot(4, 2, 2*i)
    plt.plot(f, X)
    plt.title(f'{i}. {title} – widmo amplitudowe')
    plt.xlabel('Częstotliwość [Hz]')
    plt.ylabel('|X(f)|')
    plt.grid(True)

plt.tight_layout()
>>>>>>> 60062a8918de881a9a2a2f976e270a7772a7747d
plt.show()