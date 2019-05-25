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


