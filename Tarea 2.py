import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spint
import pandas as pds
import os


################## Ejercicio 1 ############################

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
resultados_1 = RungeKutta4(f, h, n, y0, p0)


################## Ejercicio 2 ############################


#Se definen los parámetros para la aproximación.
h = 100.0 #Tamaño de paso de 100 m.
y0 = 0.0 #Elevación inicial de 0 m.
yf=3000.0 #Elevación final de 3000 m.
p0 = 101325.0 #Presión inicial de 101 325 Pa.

#Se calcula el valor de n según el h que se tenga para poder establecer un arreglo con los
# valores de y equidistantes, ese espaciamiento siendo h.
n=int((yf-y0)/h) #Recordar que n debe ser un entero, por ello el int()

#Se define el universo de valores de y de interés en que se resolverá la EDO respectiva,
#tomando en cuenta N=n+1
y = np.linspace(y0, yf, n+1) #n representa la cantidad de subintervalos, por ello el +1

#Se definen las constantes necesarias para realizar el cálculo de la función
g = 9.8 #La gravedad.
M = 0.0289647 #La masa molar en kg/mol, esto se hace para que las unidades calcen,
              #ya que el valor dado está en g/mol.            
R = 8.314462  #La constante de los gases ideales.

#Primeramente, se define la función con la que se va a trabajar, que corresponde
#a la derivada de la función que se quiere encontrar.

def F(y,P):
    '''
    Esta función corresponde al lado derecho de la EDO de primer orden P'=f(P,y).
    ----------
    y : En este caso y corresponde a la variable sobre la que se 
        desarrolla la EDO.
    P : Corresponde a la función dependiente de la variable y.

    Returns
    -------
    Salida de la función f(P,y) evaluada en y
    '''
    return -g*M*P/(R*(293.0-y/200.0)) #El resultado de la función, esto es igual a dP/dy 
                                      #en el punto dado.

#Se calculan los valores aproximados de P(y) utilizando RK45 de Scipy, los cuales se 
#almacenan en una variable.
resultados_2=(spint.solve_ivp(F, [y0, yf], [p0],t_eval=y,method='RK45')).y[0]

#Se calculan los valores verdaderos de P(y)
resultados_verdadero = (101325./(58600**(200.*g*M/R)))*(58600. - y)**(200.*g*M/R)

################## Resultados ############################

# Se prepara un gráfico para comparar los resultados
fig = plt.figure()
plt.plot(y, resultados_verdadero, 'r-', label='Valor verdadero')
plt.plot(resultados_1[0,:], resultados_1[1,:], 'b.-', label='Valor aproximado RK4')
plt.plot(y, resultados_2, 'g.-', label='Valor aproximado RK45')
plt.axis([y0, yf, resultados_2[-1], p0+1000.0])
plt.xlabel("y (m)")
plt.ylabel("P (Pa)")
plt.grid(True)
plt.title("Solucion de $dP/dy$ con $ P(0)=101325.0 Pa$")
plt.legend(loc='upper right')

# Se genera la estructura de datos para presentar los resultados
conjuntoDatos = pds.DataFrame({'Altura': y,'Aprox. RK4': np.rot90(resultados_1, k=3, axes=(0, 1))[:,0],'Aprox. RK45': resultados_2 , 'Valor Verdadero': resultados_verdadero},
                               columns=['Altura', 'Aprox. RK4', 'Aprox. RK45', 'Valor Verdadero' ])

#Se guarda el resultado en un archivo CSV al directorio actual de trabajo
conjuntoDatos.to_csv(os.path.join(os.getcwd(), 'resultados.csv'), index = False, header=True)

#Se muestran los datos
print(" ")
print(conjuntoDatos)
plt.show()


