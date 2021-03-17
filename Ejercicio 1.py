import matplotlib.pyplot as plt
import numpy as np

#fp es la función f(y,t) que representa la derivada de la función que se desea encontrar en la equación diferencial, h es el tamaño de los pasos, n es el número de pasos, x0 es el valor inicial de la variable independiente and y0 el valor de la variable dependiente (la función que se desea encontrar) en x0.
def RungeKutta4(fp, h, n, x0, y0):
    results = np.array([[x0],[y0]]) #Se inicia un arreglo para guardar los puntos calculados de la solución aproximada.
    for i in range(0, n): #Se itera sobre los pasos deseados para calcular en cada uno los puntos aproximados de la solución.
        #Se calcula cada una de las k con los valores del punto calculado en el paso anterior.
        k1 = fp(results[0, i], results[1, i])
        k2 = fp(results[0, i] + h/2, results[1, i] + h*k1/2)
        k3 = fp(results[0, i] + h/2, results[1, i] + h*k2/2)
        k4 = fp(results[0, i] + h, results[1, i] + h*k3)
        #Finalmente, se calcula el nuevo punto de la solución y se agrega al arreglo. El parámetro axis=1 hace que el nuevo punto se concatene como una columna. De esta forma, la fila de arriba son los valores de la variable independiente y los de la fila de abajo los de la variable dependiente.
        results = np.append(results,[[results[0, i] + h], [results[1, i] + (h/6)*(k1 + 2*k2 + 2*k3 + k4)]], axis=1)
    return results #Retorna el arreglo con todos los puntos calculados.

#Se define la función que es igual al valor de la derivada de la función que se desea encontrar.
def f(y,P):
    g = 9.8 #La gravedad.
    M = 0.0289647 #La masa molar en kg/mol, esto se hace para que las unidades calcen, ya que el valor dado está en g/mol.
    R = 8.314462  #La constante de gas.
    return -g*M*P/(R*(293-y/200)) #El resultado de la función, esto es igual a dP/dy en el punto dado.


#Parámetros para la aproximación.
h = 100 #Tamaño de paso de 100 m.
n = 30 #30 pasos.
y0 = 0 #Elevación inicial de 0 m.
p0 = 101325 #Presión inicial de 101 325 Pa.

#Se calculan los resultados y se grafican.
results = RungeKutta4(f, h, n, y0, p0)

plt.plot(results[0,:], results[1,:])
plt.title("P(Pa) vs. y(m)")
plt.xlabel("y(m)")
plt.ylabel("P(Pa)")
plt.show()