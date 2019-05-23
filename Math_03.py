#%%
import numpy as np

# get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

#%% [markdown]
# ## Задания к теме 3

#%% [markdown]
# #### Нарисуйте график функции: y(x) = k∙cos(x – a) + b
#
# для некоторых (2-3 различных) значений параметров k, a, b


#%%
x = np.linspace(-np.pi, 10*np.pi, 201)
k1 = 1
a1 = 1
b1 = 1
for i in range(3):
    k1 -= .25
    a1 *= .5
    b1 += .3
    plt.plot(x, k1 * np.cos(x - a1)+b1)

plt.xlabel('x')
plt.ylabel('y')
plt.axis('tight')
plt.show()

#%% [markdown]
# #### 3. Задание (в программе)
# Напишите код, который будет переводить полярные координаты в декартовы.
#
# Формулы преобразования:
# X = P*cos(A)
#
# Y = P*sin(A)

#%%
A_ = input("Введите полярный угол в градусах: ")
P_ = input("Введите полярный радиус: ")
x = int(P_) * np.cos(np.radians(int(A_)))
y = int(P_) * np.sin(np.radians(int(A_)))
print("X = ", x)
print("Y = ", y)

#%% [markdown]
# Напишите код, который будет рисовать график окружности в полярных координатах.
#%%
x = np.linspace(0, 8, 100)
y = np.linspace(5, 5, 100)
plt.polar(x, y)
plt.ylim(0,7) 

#%% [markdown]
# #### 4. Задание (в программе)
# Решите систему уравнений:
#
# y = x2 – 1
#
# exp(x) + x∙(1 – y) = 1

#%%
x = np.linspace(-3, 4, 501)
plt.plot(x, (np.exp(x) - 1)/x+1)
plt.plot(x, x**2 - 1)
plt.xlabel('x')
plt.ylabel('y')
plt.ylim(-2,10) 
plt.xlim(-3,4)
plt.grid(True)
plt.show()

from scipy.optimize import fsolve

def equations(p):
    x, y = p
    return (y - x**2 + 1, np.exp(x) + x*(1 - y)-1)

x1, y1 =  fsolve(equations, (-3, 1))
x2, y2 =  fsolve(equations, (2, 2))

print("Решения:")
print (x1, y1)
print (x2, y2)
  
