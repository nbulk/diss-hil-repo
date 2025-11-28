import numpy as np
import matplotlib.pyplot as plt

plt.style.use('~/.config/matplotlib/ant_style.mplstyle')

def generate_zc(length, root):
    q = 0
    l_0 = np.arange(0, length, 1)
    l_1 = np.arange(1, length + 1, 1)
    if length & 1:
        y = np.exp(-1j*root*np.pi/length*np.multiply(l_0, (l_1+2*q)))
    else:
        y = np.exp(-1j*root*np.pi/length*(l_0**2))

    return y

