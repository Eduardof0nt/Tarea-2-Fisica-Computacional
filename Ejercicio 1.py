import matplotlib.pyplot as plt
import numpy as np

#fp is the function f(y,t) that represents the derivative of the differential equation, h is the step size, n is the number of steps, x0 is the strat value of the independent variable and y0 is the value of the dependent variable at x0.
def RungeKutta4(fp, h, n, x0, y0):
    results = np.array([[x0],[y0]]) #Initializes an array to hold the aproximated points of the solution.
    for i in range(0, n): #Iterates through the commanded steps to find each point of the aproximated solution
        #Calculate each k factor using the previous result stored in the results array
        k1 = fp(results[0, i], results[1, i])
        k2 = fp(results[0, i] + h/2, results[1, i] + h*k1/2)
        k3 = fp(results[0, i] + h/2, results[1, i] + h*k2/2)
        k4 = fp(results[0, i] + h, results[1, i] + h*k3)
        #Finally the next result is calculated and the x and y values are added to the array. the axis=1 statement makes the new values be appended to the array as a new column (because the top row are the x's and the bottom the y's)
        results = np.append(results,[[results[0, i] + h], [results[1, i] + (h/6)*(k1 + 2*k2 + 2*k3 + k4)]], axis=1)
    return results #Returns the complete array of aproximated solutions.

#Define the function to aproximate
def f(y,P):
    g = 9.8 #The gravity
    M = 0.0289647 #Molar mass in kg/mol, this is done for matching units in the funciton, given value is 28.9647 g/mol
    R = 8.314462  #The gas constant
    return -g*M*P/(R*(293-y/200)) #The result of the function, this equals dP/dy in the given point


#Aproximation parameters
h = 100 #Step size 100 m
n = 30 #30 steps
y0 = 0 #Initial elevation of 0 m
p0 = 101325 #Initial preassure of 101 325 Pa

#The results are calculated and plotted
results = RungeKutta4(f, h, n, y0, p0)
plt.plot(results[0,:], results[1,:])
plt.title("P(Pa) vs. y(m)")
plt.xlabel("y(m)")
plt.ylabel("P(Pa)")
plt.show()