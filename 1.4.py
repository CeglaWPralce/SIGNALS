<<<<<<< HEAD
import numpy as np
import matplotlib.pyplot as plt

C=1
a=-0.01
fi = np.pi / 4
om = 0.5
Tmax = 100

t = np.linspace(-Tmax, Tmax, num=1000)

x = C * np.exp(a*t) * np.cos(om * t + fi)
y= C * np.exp(a*t)

plt.plot(t, x, 'b-', label='x(t) = A cos(ωt + φ)')
plt.plot(t, y, 'r-', label='x(t) = A cos(ωt + φ)')
plt.xlabel('t (s)')
plt.ylabel('x(t)')
plt.title('x(t) = A cos(ωt + φ)')
plt.grid(True)
plt.legend()
plt.show()

=======
import numpy as np
import matplotlib.pyplot as plt

C=1
a=-0.01
fi = np.pi / 4
om = 0.5
Tmax = 100

t = np.linspace(-Tmax, Tmax, num=1000)

x = C * np.exp(a*t) * np.cos(om * t + fi)
y= C * np.exp(a*t)

plt.plot(t, x, 'b-', label='x(t) = A cos(ωt + φ)')
plt.plot(t, y, 'r-', label='x(t) = A cos(ωt + φ)')
plt.xlabel('t (s)')
plt.ylabel('x(t)')
plt.title('x(t) = A cos(ωt + φ)')
plt.grid(True)
plt.legend()
plt.show()

>>>>>>> 60062a8918de881a9a2a2f976e270a7772a7747d
