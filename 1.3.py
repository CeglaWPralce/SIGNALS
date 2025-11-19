import numpy as np
import matplotlib.pyplot as plt

A = 1
fi = np.pi / 4
om = 1
Tmax = 10

t = np.linspace(-Tmax, Tmax, num=1000)

# x(t) = A * cos(om * t + fi)
x = A * np.cos(om * t + fi)
y= x = (A/2) * np.exp(1j * (om * t + fi)) + (A/2) * np.exp(-1j * (om * t + fi))

plt.plot(t, x, 'b-', label='x(t) = A cos(ωt + φ)')
plt.plot(t, y, 'r-', label='x(t) = A cos(ωt + φ)')
plt.xlabel('t (s)')
plt.ylabel('x(t)')
plt.title('x(t) = A cos(ωt + φ)')
plt.grid(True)
plt.legend()
plt.show()