import numpy as np
import matplotlib.pyplot as plt

# Oś czasu
t = np.linspace(-5, 5, 2000)
dt = t[1] - t[0]

# Definicja sygnału x(t) - impuls prostokątny
x = np.where(np.abs(t) < 1, 1, 0)

# Lista do przechowywania splotów
splines = [x]  # Pierwszy element to oryginalny sygnał x(t)

# Obliczanie kolejnych splotów
for i in range(4):
    # Oblicz splot poprzedniego sygnału z x(t)
    next_spline = np.convolve(splines[-1], x, mode='same') * dt
    splines.append(next_spline)

# Wykres
plt.figure(figsize=(10, 6))

# Rysowanie oryginalnego sygnału i splotów
colors = ['b', 'r', 'g', 'm', 'c']  # Różne kolory dla każdego sygnału
labels = ['x(t)', 'Splot 1', 'Splot 2', 'Splot 3', 'Splot 4']

for i in range(5):
    plt.plot(t, splines[i], colors[i], label=labels[i])

# Dodanie tytułu, etykiet i legendy
plt.title('Oryginalny sygnał x(t) i kolejne sploty')
plt.xlabel('t')
plt.ylabel('Amplituda')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()