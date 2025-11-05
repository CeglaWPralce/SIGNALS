import numpy as np
import matplotlib.pyplot as plt

# Parametry
k = 0.5
fi = np.pi / 4
omm = 1
om0 = 10
Tmax = 10

# Oś czasu
t = np.linspace(-Tmax, Tmax, num=1000)

# Sygnał z modulacją fazy
x = np.cos(om0 * t + k * np.cos(omm * t))

# Częstotliwość chwilowa (pochodna fazy względem czasu)
y = om0 - k * omm * np.sin(omm * t)

# Rysowanie
plt.figure(figsize=(10, 6))
plt.plot(t, x, 'b-', label='x(t) = cos(ω₀t + k cos(ωₘt))')
plt.plot(t, y, 'r--', label="ω_inst(t) = ω₀ - kωₘ sin(ωₘt)")
plt.xlabel('t (s)')
plt.ylabel('Wartość')
plt.title('Sygnał z modulacją fazy i jego częstotliwość chwilowa')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()