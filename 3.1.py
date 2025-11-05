<<<<<<< HEAD
import numpy as np
import matplotlib.pyplot as plt


T0 = 0.5
T1 = 0.125
N = 10
pi = np.pi

t = np.linspace(-T0, T0, 2000)
x = np.where(np.abs(t) < T1, 1, 0)


a_k = []
k_vals = np.arange(-N, N + 1)
for k in k_vals:
    if k == 0:
        a_k.append(2 * T1 / T0)
    else:
        a_k.append(np.sin(k * pi * T1 / T0) / (k * pi))

a_k = np.array(a_k)


xN = np.zeros_like(t, dtype=complex)
for k, ak in zip(k_vals, a_k):
    xN += ak * np.exp(1j * 2 * pi * k * t / T0)
xN = np.real(xN)


plt.figure(figsize=(10, 6))
plt.plot(t, x, 'k-', label='Sygnał oryginalny (prostokątny)')
plt.plot(t, xN, 'r--', label=f'Aproksymacja szereg. Fouriera (N={N})')
plt.xlabel('t [s]')
plt.ylabel('Amplituda')
plt.title('Zadanie 3.1 – Aproksymacja sygnału prostokątnego szeregiem Fouriera')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 4))
plt.stem(k_vals, a_k)
plt.xlabel('k')
plt.ylabel('a_k')
plt.title('Współczynniki szeregu Fouriera')
plt.grid(True)
plt.tight_layout()
=======
import numpy as np
import matplotlib.pyplot as plt


T0 = 0.5
T1 = 0.125
N = 10
pi = np.pi

t = np.linspace(-T0, T0, 2000)
x = np.where(np.abs(t) < T1, 1, 0)


a_k = []
k_vals = np.arange(-N, N + 1)
for k in k_vals:
    if k == 0:
        a_k.append(2 * T1 / T0)
    else:
        a_k.append(np.sin(k * pi * T1 / T0) / (k * pi))

a_k = np.array(a_k)


xN = np.zeros_like(t, dtype=complex)
for k, ak in zip(k_vals, a_k):
    xN += ak * np.exp(1j * 2 * pi * k * t / T0)
xN = np.real(xN)


plt.figure(figsize=(10, 6))
plt.plot(t, x, 'k-', label='Sygnał oryginalny (prostokątny)')
plt.plot(t, xN, 'r--', label=f'Aproksymacja szereg. Fouriera (N={N})')
plt.xlabel('t [s]')
plt.ylabel('Amplituda')
plt.title('Zadanie 3.1 – Aproksymacja sygnału prostokątnego szeregiem Fouriera')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 4))
plt.stem(k_vals, a_k)
plt.xlabel('k')
plt.ylabel('a_k')
plt.title('Współczynniki szeregu Fouriera')
plt.grid(True)
plt.tight_layout()
>>>>>>> 60062a8918de881a9a2a2f976e270a7772a7747d
plt.show()