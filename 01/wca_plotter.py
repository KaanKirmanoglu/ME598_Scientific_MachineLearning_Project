import numpy as np
import matplotlib.pyplot as plt



def lj(r,sigma,eps):
    return (4*eps*(np.power(np.divide(sigma,r),12) - np.power(np.divide(sigma,r),6)))

def wca(r,sigma,eps,r_cut):
    shift = lj(r_cut,sigma,eps)
    wca = lj(r,sigma,eps) - shift
    wca[r>r_cut] = 0.0
    return wca

def get_r_cut(d1,d2):
    """
    where you need to cut LJ for larger particles interacting
    """
    sigma = (d1+d2)*0.5
    r_cut = 2.0**(1/6.)*sigma
    return r_cut

r = np.linspace(0.9, 6.0, num=100000)
dA = 4.0
dB = 1.0
cutAB = get_r_cut(dA,dB)

plt.figure(1)
plt.plot(r,wca(r,1.0,1.0,2**(1/6.)))
plt.plot(r,wca(r,2.5,1.0,cutAB))
plt.ylim(-2.0,10.0)
plt.show()














exit()
