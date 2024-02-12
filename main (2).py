import numpy as np
import matplotlib.pyplot as plt
import math

#########################################
###############FUNCTIONS#################
#########################################

def polarConverter(x,y):
  r = np.sqrt(x**2 + y**2)
  if x != 0 or y != 0:
    theta = math.atan2(y,x)
  else:
     theta = 0
  return r,theta

def laguerre(n, l, x):
    array_size = n - l
    array = np.zeros(array_size)
    for i in range(array_size):
       array[i] = (-1)**(i + 1)*math.factorial(n + l)**2/(math.factorial(n-1-l-i)*math.factorial(2*l + 1 + i)*math.factorial(i))*x**i
    L = np.sum(array)
    return L

def radial(n, l, r):
    R = (2**(l+1)) / (n**(l+2)) * np.sqrt(math.factorial(n-l-1) / ((2*n) * math.factorial(n+1))) * \
        r**l * np.exp(-r/n) * laguerre(n, l, 2*r/n)
    return R

def associatedLegendre(l, m, x):
    P = np.polynomial.legendre.Legendre.basis(l)(x)
    return P

def sphericalHarmonic(l, m, theta, phi):
    Y = np.sqrt(((2*l+1)/(4*np.pi)) * (math.factorial(l-m) / math.factorial(l+m))) * \
        associatedLegendre(l, m, np.cos(theta)) * (np.cos(phi*m) + 1j*np.sin(phi*m))
    return Y

def wavefunction(n, l, m, r, theta, phi):
    psi = radial(n, l, r) * sphericalHarmonic(l, m, theta, phi)
    return np.abs(psi)**2

#########################################
###############Main code#################
#########################################

n, l, m = 3, 1, 0
nlm = str(str(n)+str(l)+str(m))

xmin, xmax, ymin, ymax = -20, 20, -20, 20
step = 100
xrange = np.linspace(xmin, xmax, step)
yrange = np.linspace(ymin, ymax, step)

emptyMatrix = np.zeros((step, step))  # for storing wavefunction 

# Generating points, and thus calculating the value of the wavefunction at that point.
for i, x in enumerate(xrange):
    for j, y in enumerate(yrange):
        r, theta = polarConverter(x, y)
        phi = np.pi/2
        emptyMatrix[i, j] = wavefunction(n, l, m, r, theta, phi)

plt.imshow(emptyMatrix, interpolation="bilinear", origin="upper", cmap="hot")



plt.title(r'$\Psi_{{{}}}$'.format(nlm))
plt.axis('off')
plt.savefig(nlm+".png",bbox_inches='tight')
plt.show()