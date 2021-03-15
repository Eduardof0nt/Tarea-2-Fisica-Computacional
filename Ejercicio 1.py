import matplotlib.pyplot as plt
import numpy as np

#fp is the function f(y,t) that represents the derivative of the differential equation, h is the step size, n is the number of steps, x0 is the strat value of the independent variable and y0 is the value of the dependent variable at x0.
def RungeKutta4(fp, h, n, x0, y0):
    results = np.array([[x0],[y0]])
    for i in range(0, n+1):
        k1 = fp(results[0, i], results[1, i])
        k2 = fp(results[0, i] + h/2, results[1, i] + h*k1/2)
        k3 = fp(results[0, i] + h/2, results[1, i] + h*k2/2)
        k4 = fp(results[0, i] + h, results[1, i] + h*k3)
        results = np.append(results,[[results[0, i] + h], [results[1, i] + (h/6)*(k1 + 2*k2 + 2*k3 + k4)]], axis=1)
    return results

def f(x,y):
    g = 9.8
    M = 28.9647
    R = 8.314462 
    return -g*M*y/(R*(293-x/200))

h = 100
n = 30
y0 = 0
p0 = 101325

results = RungeKutta4(f, h, n, y0, p0)
plt.plot(results[0,:], results[1,:])