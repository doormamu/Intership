import numpy as np
import matplotlib.pyplot as plt

# Число разбиений
N = 20

# Шаг сетки
h = 1.0 / N

# Узлы сетки x_0, x_1, ..., x_N
x = np.linspace(0, 1, N + 1)


def f(x):
    """
    Правая часть дифференциального уравнения:
    f(x) = (pi^2 / 4 + x^2) * cos(pi*x/2)
    """
    return (np.pi**2 / 4 + x**2) * np.cos(np.pi * x / 2)


def u_exact(x):
    """
    Точное решение задачи:
    u(x) = cos(pi*x/2)
    """
    return np.cos(np.pi * x / 2)


# Размер системы (число неизвестных)
M = N - 1

# Матрица системы
A = np.zeros((M, M))

# Вектор правой части
b = np.zeros(M)

for i in range(M):
    xi = x[i + 1]  # соответствующий узел x_{i+1}

    # Главная диагональ
    A[i, i] = 2 + h**2 * xi**2

    # Нижняя диагональ
    if i > 0:
        A[i, i - 1] = -1

    # Верхняя диагональ
    if i < M - 1:
        A[i, i + 1] = -1

    # Правая часть
    b[i] = h**2 * f(xi)


# Учитываем u(0) = 1
b[0] += 1

# u(1) = 0, поэтому для последнего уравнения ничего добавлять не нужно

# Решаем линейную систему
u_inner = np.linalg.solve(A, b)


# Полный массив решения с учётом границ
u_num = np.zeros(N + 1)

u_num[0] = 1.0          # u(0)
u_num[1:N] = u_inner   # внутренние узлы
u_num[N] = 0.0          # u(1)


# Точное решение в узлах сетки
u_ex = u_exact(x)

# Максимальная ошибка
error = np.max(np.abs(u_num - u_ex))

print(f"Максимальная ошибка: {error:.6e}")


plt.figure(figsize=(8, 5))

plt.plot(x, u_ex, label="Точное решение", linewidth=2)
plt.plot(x, u_num, "o--", label="Численное решение (МКР)")

plt.xlabel("x")
plt.ylabel("u(x)")
plt.legend()
plt.grid(True)

plt.show()


