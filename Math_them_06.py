#%%
import numpy as np
# get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

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
# #### Решение переопределенной СЛАУ
#### $x + 2y -z = 4$
#### $3x – 4y = 7$
#### $8x – 5y + 2z = 12$
#### $2x – 5z = 7$
#### $11x +4y – 7z = 15$


