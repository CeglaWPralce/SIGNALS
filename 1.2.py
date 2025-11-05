<<<<<<< HEAD
import numpy as np
import matplotlib.pyplot as plt

Om = 1           # Częstotliwość kątowa
Tmax = 10

t = np.linspace(-Tmax, Tmax, num=1000)  # więcej punktów dla gładkości
x = np.exp(1j * Om * t)                 # poprawnie: exp(j*Om*t)

# Rysujemy osobno część rzeczywistą i urojoną
plt.plot(t, np.real(x), 'b-', label='Re{x(t)} = cos(Ωt)')
plt.plot(t, np.imag(x), 'r-', label='Im{x(t)} = sin(Ωt)')
plt.xlabel('t (s)')
plt.ylabel('x(t)')
plt.title('x(t) = exp(jΩt)')
plt.grid(True)
=======
import numpy as np
import matplotlib.pyplot as plt

Om = 1           # Częstotliwość kątowa
Tmax = 10

t = np.linspace(-Tmax, Tmax, num=1000)  # więcej punktów dla gładkości
x = np.exp(1j * Om * t)                 # poprawnie: exp(j*Om*t)

# Rysujemy osobno część rzeczywistą i urojoną
plt.plot(t, np.real(x), 'b-', label='Re{x(t)} = cos(Ωt)')
plt.plot(t, np.imag(x), 'r-', label='Im{x(t)} = sin(Ωt)')
plt.xlabel('t (s)')
plt.ylabel('x(t)')
plt.title('x(t) = exp(jΩt)')
plt.grid(True)
>>>>>>> 60062a8918de881a9a2a2f976e270a7772a7747d
