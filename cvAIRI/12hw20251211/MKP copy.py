import numpy as np
import matplotlib.pyplot as plt



N = 20
h = 1.0 / N
x = np.array([i * h for i in range(N + 1)])


def f(x):
    return (np.pi**2 / 4 + x**2) * np.cos(np.pi * x / 2)

def u_exact(x):
    return np.cos(np.pi * x / 2)


M = N - 1

a = np.zeros(M) 
b = np.zeros(M)  
c = np.zeros(M)  
d = np.zeros(M)  

for i in range(M):
    xi = x[i + 1]

    a[i] = -1.0
    b[i] = 2.0 + h**2 * xi**2
    c[i] = -1.0
    d[i] = h**2 * f(xi)


d[0] += 1.0      # u(0) = 1
c[M - 1] = 0.0   # u(1) = 0


alpha = np.zeros(M)
beta = np.zeros(M)

alpha[0] = c[0] / b[0]
beta[0] = d[0] / b[0]

for i in range(1, M):
    denom = b[i] - a[i] * alpha[i - 1]
    alpha[i] = c[i] / denom if i < M - 1 else 0.0
    beta[i] = (d[i] - a[i] * beta[i - 1]) / denom

u_inner = np.zeros(M)
u_inner[M - 1] = beta[M - 1]

for i in range(M - 2, -1, -1):
    u_inner[i] = beta[i] - alpha[i] * u_inner[i + 1]


u_num = np.zeros(N + 1)
u_num[0] = 1.0
u_num[1:N] = u_inner
u_num[N] = 0.0

u_ex = u_exact(x)


print(" i |    x_i    |   u_i (числ.)   |   u_i (точн.)   |   ошибка")
print("-" * 70)

for i in range(N + 1):
    err = abs(u_num[i] - u_ex[i])
    print(f"{i:2d} | {x[i]:8.5f} | {u_num[i]:14.8f} | {u_ex[i]:14.8f} | {err:10.2e}")

print("-" * 70)


plt.plot(x, u_ex, label="Точное решение")
plt.plot(x, u_num, "o--", label="Численное")
plt.xlabel("x")
plt.ylabel("u(x)")
plt.legend()
plt.grid(True)
plt.show()
