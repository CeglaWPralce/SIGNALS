

import numpy as np
import matplotlib.pyplot as plt

Tmax = 1
N = 10  # mniejsza liczba, żeby było widać kształty

t = np.linspace(-Tmax, Tmax, num=N)
x = -t*t + 0.5*t + 2
dt = t[1] - t[0]

# Metoda prostokątów (lewe końce)
IR = np.sum(x[0:N-1]) * dt

# Metoda trapezów
IT = np.sum((x[0:N-1] + x[1:N]) / 2) * dt

print('IR =', IR)
print('IT =', IT)

# Siatka gęsta do rysowania krzywej
t_plot = np.linspace(-Tmax, Tmax, 1000)
x_plot = -t_plot**2 + 0.5*t_plot + 2

plt.figure(figsize=(10, 6))
plt.plot(t_plot, x_plot, 'k-', label='f(t) = -t² + 0.5t + 2')

# --- Prostokąty (niebieskie) ---
for i in range(N - 1):
    plt.bar(t[i], x[i], width=dt, align='edge', alpha=0.3, edgecolor='blue', color='skyblue')

# --- Trapezy (zielone) ---
for i in range(N - 1):
    xs = [t[i], t[i+1], t[i+1], t[i]]
    ys = [0, 0, x[i+1], x[i]]
    plt.fill(xs, ys, color='lightgreen', alpha=0.3, edgecolor='green')

plt.xlabel('t (s)')
plt.ylabel('x(t)')
plt.title(f'Porównanie metod całkowania (N = {N})')
plt.legend(['f(t)', 'Prostokąty', 'Trapezy'])
plt.grid(True)
plt.show()
