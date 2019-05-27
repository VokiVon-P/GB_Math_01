#%%
import numpy as np
# get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import scipy 
import scipy.linalg as ln

#%% [markdown]
# ## Задания к теме 6

#%%

#%%
A = np.array([[1, 2, 3], [4, 0, 6], [7, 8, 9]])
B = np.array([12, 2, 1])
X = np.linalg.solve(A, B)
print("Решение:\n", X)

#%% [markdown]
# Решение через обратную матрицу
# X = inv(A)⋅B
#
# Решение возможно только если определитель A != 0
#%%
dA = np.linalg.det(A)
print("det =", dA)
if dA != 0:
    print("Можем найти решение!")
    print("Обратная матрица:")
    A1 = np.linalg.inv(A)
    print(A1)
    X1 = np.dot(A1, B)

    print("Решение:\n", X1)
else:
    print("Система не имеет решения")


#%% [markdown]
# #### 2 - Решение переопределенной СЛАУ
#### $x + 2y -z = 4$
#### $3x – 4y = 7$
#### $8x – 5y + 2z = 12$
#### $2x – 5z = 7$
#### $11x +4y – 7z = 15$

#%%
A = np.array([[1, 2, -1], [3, -4, 0], [8, -5, 2], [2, -5, 0], [11, 4, -7]])
B = np.array([4, 7, 12, 7, 15])
res = np.linalg.lstsq(A, B)
print(res)
print("Псевдорешение:\n",res[0])
print("Сумма квадратов отклонений(невязок): = ", res[1][0])
print("Ранк = ",res[2])

#%% [markdown]
# #### 3 - Решение  квадратной матрицы

#%%
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
B = np.array([[12, 2, 1]])
C = np.concatenate((A,B.T), axis=1)
print("A:\n",A)
print("C:\n",C)
print("Rank A = ", np.linalg.matrix_rank(A, 0.0001))
print("Rank C = ", np.linalg.matrix_rank(C, 0.0001))
print("Так как Rank A < Rank C то данная СЛАУ не имеет решения!")

#%%
print("Изменим систему: A = [[1, 2, 3], [4, 10, 6], [7, 8, 9]]")
A = np.array([[1, 2, 3], [4, 10, 6], [7, 8, 9]])
B = np.array([[12, 2, 1]])
C = np.concatenate((A,B.T), axis=1)
print("A:\n",A)
print("C:\n",C)
print("Rank A = ", np.linalg.matrix_rank(A, 0.0001))
print("Rank C = ", np.linalg.matrix_rank(C, 0.0001))
print("Так как Rank A = Rank С и = числу переменных то система имеет единственное решение!")
X = np.linalg.solve(A, B[0])
print("Решение:\n", X)

#%% [markdown]
# #### 4 - Решение  методом LU разложения ( методм Гаусса )

#%%
A = np.array([ [1, 2, 3], [2, 16, 21], [4, 28, 73] ])
B = np.array([12, 2, 1])
P, L, U = ln.lu(A)
print("det A = ", np.linalg.det(A))
print("P = ", P)
print("L = ", L)
print("U = ", U)

print("L*Y = B")
Y = np.linalg.solve(L, B)
print("Y = ", Y)

print("U*X = Y")
X = np.linalg.solve(U, Y)
print("X = ", X)


#%% [markdown]
# #### 5. Найдите нормальное псевдорешение недоопределенной системы:
#
# x + 2y – z = 1
#
# 8x – 5y + 2z = 12
#
# Для этого определите функцию Q(x,y,z), равную норме решения, и найдите ее минимум. 
#%%
A = np.array([ [1, 2, -1], [8, -5, 2] ])
B = np.array([1, 12])

def Q(x, y, z):
    return (x**2 + y**2 + z**2)

x = np.linspace(-1, 4, 301)
Y = 10*x - 14
Z = x + 2*Y - 1

#%%
# минимум по X
plt.plot(x, Q(x, Y, Z))
plt.xlabel('x')
plt.ylim(0,10) 
plt.xlim(1.2,1.5)
plt.grid(True)
plt.show()
print("Минимум по X находиться между 1.35 и 1.4")

#%%
# минимум по Y
plt.plot(Y, Q(x, Y, Z))
plt.xlabel('y')
plt.ylim(0,10) 
plt.xlim(-1, 1)
plt.grid(True)
plt.show()
print("Минимум по Y находиться между 0 и -0.25")

#%%
# минимум по Z
plt.plot(Z, Q(x, Y, Z))
plt.xlabel('z')
plt.ylim(0,10) 
plt.xlim(-2, 2)
plt.grid(True)
plt.show()
print("Минимум по Z находиться в районе 0 ")

#%%
res = np.linalg.lstsq(A, B)
print("Алгебраическое решение: X = ", res[0][0], " Y = ", res[0][1], " Z = ", res[0][2])

#%%
""" from pylab import *
from mpl_toolkits.mplot3d import Axes3D
fig = figure()
ax = Axes3D(fig)
ax.set_xlim(0, 2)
ax.set_ylim(-1, 1)
ax.set_zlim(0,12)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

x = np.linspace(-1, 4, 301)
Y = 10*x - 14
Z = 21*x - 28
# Z = Q(x, 10*x -14, 21*x - 28)
# ax.plot_surface(X, Y, Z)
ax.plot(x, Y, Q(x, Y, Z))
ax.zaxis.axis_name = "hello"
show() """

#%% [markdown]
# #### 6. Найдите одно из псевдорешений вырожденной системы:

#%%
A = np.array([ [1, 2, 3], [4, 5, 6], [7, 8, 9] ])
B = np.array([2, 5, 11])
Q, R = np.linalg.qr(A)

print(A)
print(Q)
print(R)
print("\nПроверка полученных матриц:")
print(np.dot(Q, R))
print(np.dot(Q.T, Q))

#%%
print("\nСоздадим R1 и B1:")
R1 = R[:2, :2]
print(R1)
B1 = np.dot(Q.T, B)[:2]
print(B1)

print("\nПолучим решение и склеим с X2 = 0:")
X1 = np.linalg.solve(R1, B1)
print(X1)
X3 = np.append(X1, 0)
print("\nОдно из псевдорешений:")
print (X3)
print("\nс нормой = ", np.linalg.norm(X3))
np.linalg.norm(X3),  np.linalg.norm(np.dot(A, X3) - B)      

#%% [markdown]
# Ищем решение с минимальной нормой
#%%
res = np.linalg.lstsq(A, B, rcond=None)
print(res)
X = res[0]
print("\nАлгебраическое решение: ", X)


print("\nс нормой = ", np.linalg.norm(X))
np.linalg.norm(X),  np.linalg.norm(np.dot(A, X) - B)  