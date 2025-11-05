<<<<<<< HEAD
import numpy as np
import matplotlib.pyplot as plt

a=-1
C=1e4
Tmax = 100
t = np.linspace(-Tmax, Tmax, num=2*Tmax+1)
x = C*np.exp(a*t)

plt.plot(t,x,'b.-')
plt.xlabel('t(s)')
plt.ylabel('x(t)')
plt.title('x(t)=C*exp(at)')
plt.legend([ 'a=' + str(a) + ', C=' + str(C)])
plt.grid(True)
plt.show()
=======
import numpy as np
import matplotlib.pyplot as plt

a=-1
C=1e4
Tmax = 100
t = np.linspace(-Tmax, Tmax, num=2*Tmax+1)
x = C*np.exp(a*t)

plt.plot(t,x,'b.-')
plt.xlabel('t(s)')
plt.ylabel('x(t)')
plt.title('x(t)=C*exp(at)')
plt.legend([ 'a=' + str(a) + ', C=' + str(C)])
plt.grid(True)
plt.show()
>>>>>>> 60062a8918de881a9a2a2f976e270a7772a7747d
