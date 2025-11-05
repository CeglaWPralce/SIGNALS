import numpy as np
import matplotlib.pyplot as plt

# Oś czasu
t = np.linspace(-5, 5, 2000)
dt = t[1] - t[0]

# Definicje sygnałów x(t) i h(t)
x = np.where(np.abs(t) < 1, 0.5, 0)
h = np.where(np.abs(t) < 1, t**2 + 1, 0)

# Splot całkowy (numerycznie)
y = np.convolve(x, h, mode='same') * dt

# Wykresy
plt.figure(figsize=(10, 8))

plt.subplot(3, 1, 1)
plt.plot(t, x, 'b')
plt.title('x(t)')
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(t, h, 'r')
plt.title('h(t)')
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(t, y, 'k')
plt.title('y(t) = x(t) * h(t)')
plt.grid(True)

plt.tight_layout()
plt.show()


