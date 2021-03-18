# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spint
import pandas as pds

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
p_aprox_RK45=(spint.solve_ivp(F, [y0, yf], [p0],t_eval=y,method='RK45')).y[0]

#Se calculan los valores verdaderos de P(y)
p_verdadero = (101325./(58600**(200.*g*M/R)))*(58600. - y)**(200.*g*M/R)

# Se prepara un gráfico para comparar los resultados
fig = plt.figure()
plt.plot(y, p_verdadero, 'r-', label='Valor verdadero')
plt.plot(y, p_aprox_RK45, 'g.-', label='Valor aproximado RK45')
plt.axis([y0, yf, p_aprox_RK45[-1], p0+1000.0])
plt.xlabel("y (m)")
plt.ylabel("P (Pa)")
plt.grid(True)
plt.title("Solucion de $dP/dy$ con $ P(0)=101325.0 Pa$")
plt.legend(loc='upper right')

# Se genera la estructura de datos para presentar los resultados
conjuntodatos = pds.DataFrame({'altura': y, 'Aprox. RK45': p_aprox_RK45 , 'Valor verdadero': p_verdadero},\
                               columns=['altura', 'Aprox. RK45', 'Valor verdadero' ])
print(" ")
print(conjuntodatos)
    
    
    
    
    
    
    
    
    
    
    
    
