import numpy as np
import matplotlib.pyplot as plt

def dft_matrix(n):
	dft_mat = np.zeros((n,n),dtype=np.complex128)
	for i in range(n):
		for j in range(n):
				dft_mat[i][j] = np.exp(-2j*np.pi*i*j/n)
	return dft_mat

def dft(x):
	n = len(x)
	F = dft_matrix(n)
	return F@x

def h_n(N):
    h = []
    for i in range(N):
        o = 0
        if i >= 0:
            o += pow(-0.5, i)
        if i - 2 >= 0:
            o += pow(-0.5, i - 2)
        h.append(o)
    return h

N = 6
x = np.array([1,2,3,4,2,1])
X = dft(x)
h = h_n(N)
H = dft(h)
Y = np.multiply(X,H)
plt.figure(1,figsize=(9,7.5))
plt.subplot(3,2,1)
plt.stem(np.abs(X),use_line_collection=True)
plt.title(r'$|X(k)|$')
plt.grid()

plt.subplot(3,2,2)
plt.stem(np.angle(X),use_line_collection=True)
plt.title(r'$\angle{X(k)}$')
plt.grid()

plt.subplot(3,2,3)
plt.stem(np.abs(H),use_line_collection=True)
plt.title(r'$|H(k)|$')
plt.grid()

plt.subplot(3,2,4)
plt.stem(np.angle(H),use_line_collection=True)
plt.title(r'$\angle{H(k)}$')
plt.grid()


plt.subplot(3,2,5)
plt.stem(np.abs(Y),use_line_collection=True)
plt.title(r'$|Y(k)|$')
plt.grid()

plt.subplot(3,2,6)
plt.stem(np.angle(Y),use_line_collection=True)
plt.title(r'$\angle{Y(k)}$')
plt.grid()


plt.show()
