#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 12:40:29 2019

@author: José Manuel Pérez Lendínez
"""
import random
import numpy as np
from scipy.io import arff
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics.pairwise import euclidean_distances
import sys
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from time import time
from sklearn.neighbors import KDTree
semilla = 1
np.random.seed(semilla)
#rutas de los ficheros
colopscopy = "datos/colposcopy.arff"
ionosphere = "datos/ionosphere.arff"
texture = "datos/texture.arff"

#Peso apartir el cual no se tendra en cueta la caracteristica
min_peso = 0.2

#numero de particiones para los datos
num_particiones = 5

#max_veinos que obtendremos
max_vecinos = 15000

#valor para la distribucion de la mutacion
sigma = 0.3

tam_poblacion = 30
num_evaluaciones = 15000
porcentaje_cruce = 0.7
porcentaje_mutacion = 0.001
#Calcula la funcion objetivo
def funcionObjetivo(clase,reduccion):
    return clase * 0.5 + reduccion * 0.5

#algoritmos de relief
def RELIEF(training, test):
    
  
    #nos quedamos con los datos del training si las clases
    train_datos = training[: , 0:-1];
    num_elementos = np.size(training,0)
    
    start_time = time()
    
    #iniciamos el vector de pesos a 0
    num_valores = training[0].size - 1
    w = np.zeros(num_valores)
    
    #creo una matriz de distancias
    distancias = euclidean_distances(train_datos, train_datos)
    #print(distancias)
    
    #recorro las columnas de la matriz de distancia
    for i in range(num_elementos):
        #inicializo los mejores valores de vecino y enemigo con el maximo
        mejor_enemigo = int()
        valor_enemigo = sys.float_info.max
        mejor_amigo = int()
        valor_amigo = sys.float_info.max
        #recorremos las filas para mirar el mejor amigo y enemigo
        for j in range(num_elementos):
            #Si la clase es igual se mira si no es si se mejora y en caso de no
            #si mismo se almacena como mejor amigo
            if training[i][-1] == training[j][-1]:
                if valor_amigo > distancias[i][j]:
                    if i != j:
                        valor_amigo = distancias[i][j]
                        mejor_amigo = j
            
            #si mejora nos quedamos con el pero enemigo
            else:
                if valor_enemigo > distancias[i][j]:
                    valor_enemigo = distancias[i][j]
                    mejor_enemigo = j
        
        #actualizamos el vector 
        w = w + np.absolute(train_datos[i]-train_datos[mejor_enemigo]) -\
        np.absolute(train_datos[i]-train_datos[mejor_amigo])
    
    #nos quedamos el mayor elemento del vector de pesos
    w_maximo = np.amax(w)
    
    #normalizamos el vector de pesos
    for i in range(num_valores):
        if w[i] < 0:
            w[i] = 0.0
        else:
            w[i] = w[i] / w_maximo
    
    #Nos quedamos con las clases y  los datos por separado
    train_clases = np.array(training[:, -1], int)
    test_datos = test[:, 0:-1]
    test_clases = np.array(test[:, -1], int)
    
    #tasa_clase, tasa_reduccion = uno_nn(train_datos,train_clases,test_datos,test_clases,w)
    tasa_clase, tasa_reduccion, funcion_mejora = evaluate(w, train_datos, train_clases)
    
    
    datos_medidos = np.zeros(4)
    
    #almacenamos los datos
    datos_medidos[3] = time() - start_time  
    datos_medidos[0] = tasa_clase
    datos_medidos[1] = tasa_reduccion
    datos_medidos[2] = funcionObjetivo(datos_medidos[0], datos_medidos[1])
    
    return datos_medidos

"""
Utilzado para el 1-NN donde no tenemos vector de pesos
"""

def k_nn(training,test):
    #nos quedmoas con los datos y las clases por separado
    train_datos = training[:, 0:-1]
    train_clases = np.array(training[:, -1], int)
    test_datos = test[:, 0:-1]
    test_clases = np.array(test[:, -1], int)
    
    start_time = time()
    num_aciertos = 0;
    
    #preparamos el clasificador con el train
    clasificador = KNeighborsClassifier(n_neighbors=1)
    clasificador.fit(train_datos, train_clases)
    num_muestras = np.size(test_datos,0)
    
    #clasificamos con el test
    for i in range(num_muestras):
        tipo = clasificador.predict([test_datos[i]])
        if( tipo == test_clases[i]):
            num_aciertos += 1
    tasa_acierto = 100 * (num_aciertos / num_muestras)
    tiempo = time() - start_time  
    funcion = funcionObjetivo(tasa_acierto,0)
    
    datos_algoritmo = np.zeros(4)
    datos_algoritmo[0] = tasa_acierto
    datos_algoritmo[2] = funcion
    datos_algoritmo[3] = tiempo
    
    return datos_algoritmo

def evaluate(weights, X, y):
    X_transformed = (X * weights)[:, weights > 0.2]
    kdtree = KDTree(X_transformed)
    neighbours = kdtree.query(X_transformed, k=2)[1][:, 1]
    accuracy = np.mean(y[neighbours] == y)
    reduction = np.mean(weights < 0.2)
    return 100*accuracy,reduction*100,100*(accuracy + reduction) / 2

def uno_nn_antiguo(train_datos, train_clases, test_datos, test_clases, w):
    
    
    #nos quedamos con los datos que cumplen el minimo para los pesos
    train_datos = (train_datos * w)[: , w >= min_peso]
    test_datos = (test_datos * w)[: , w >= min_peso]   

    num_aciertos = 0;
    
    #entrenamos con el train 
    clasificador = KNeighborsClassifier(n_neighbors=1)
    clasificador.fit(train_datos, train_clases)
    num_muestras = np.size(test_datos,0)
    
    #clasificamos con el test
    for i in range(num_muestras):
        tipo = clasificador.predict([test_datos[i]])
        if( tipo == test_clases[i]):
            num_aciertos += 1
            
    tasa_acierto = 100 * (num_aciertos / num_muestras)
    tasa_reduccion = 100 * (w[w < min_peso].size / w.size)
    
    return tasa_acierto,tasa_reduccion

def uno_nn(train_datos, train_clases, test_datos, test_clases, w):
    
    
    #nos quedamos con los datos que cumplen el minimo para los pesos
    train_datos = (train_datos * w)[: , w >= min_peso]
    test_datos = (test_datos * w)[: , w >= min_peso]   

    num_aciertos = 0;
    
    #entrenamos con el train 
    clasificador = KNeighborsClassifier(n_neighbors=1)
    clasificador.fit(train_datos, train_clases)
    num_muestras = np.size(train_datos,0)
    
    
    
    #clasificamos con el test
    for i in range(num_muestras):
        tipo = clasificador.predict([train_datos[i]])
        if( tipo == train_clases[i]):
            num_aciertos += 1
            
    tasa_acierto = 100 * (num_aciertos / num_muestras)
    tasa_reduccion = 100 * (w[w < min_peso].size / w.size)
    
    return tasa_acierto,tasa_reduccion


def BL(training,test):
    
    
    #Preparamos los datos
    train_datos = training[:, 0:-1]
    train_clases = np.array(training[:, -1], int)
    test_datos = test[:, 0:-1]
    test_clases = np.array(test[:, -1], int)
    
    #Contador de vecionos creados
    num_vecinos = 0
    
    num_valores = train_datos[0].size
    
    #inciamos el vector con valores random
    w = np.random.rand(num_valores)

    
    #marcara la posicion a mutar paa el vector
    pos_w = 0
    
    #contador de veces in obtener mejora y el maximo de datos sin mejora posible
    sin_mejora = 0
    max_sin_mejora = 20 * num_valores
    
    
    start_time = time()
    
    #iniciamos los valores para el primer vector 
    #tasa_clase, tasa_reduccion = uno_nn(train_datos, train_clases, test_datos, test_clases, w)
    tasa_clase, tasa_reduccion, funcion_mejora = evaluate(w, train_datos, train_clases)
            
    #funcion_mejora = 0.5 * tasa_clase + 0.5 * tasa_reduccion
    mejor_valor_w = funcion_mejora
    mejor_w = w
    mejor_tasa_clase = tasa_clase
    mejor_tasa_reduccion = tasa_reduccion

    
    while num_vecinos < max_vecinos and sin_mejora < max_sin_mejora:
        
        #contamos un nuevo vecion
        num_vecinos += 1
        
        #generamos la mutacion y guardamos el anterior por si no mejora
        cambio = np.random.normal(0.0, sigma, None)
        w_anterior = w[pos_w]
        
        #miramos que no pasa ni 0 ni 1
        w[pos_w] += cambio
        if w[pos_w] > 1:
            w[pos_w] = 1
        if w[pos_w] < 0:
            w[pos_w] = 0
            
        """
        Si la mutacio da un peso menor al minimo y el anterior tambien era menor
        o la mutacion deja los pesos como estaban. No podra mejorar nunca los datos
        de entrenamiento y no hace falta hacer los calculos. Se cuenta uno sin mejora.
        """
        if (w_anterior < min_peso and w[pos_w] < min_peso) or (w_anterior == w[pos_w]):
             w[pos_w] = w_anterior
             sin_mejora += 1
        #en otro caso si hay que hacer los calculos de la funcion de mejora
        else:
            
            #realizamos los calculos para ver si mejora
            #tasa_clase, tasa_reduccion = uno_nn(train_datos, train_clases, test_datos, test_clases, w)      
            tasa_clase, tasa_reduccion, funcion_mejora = evaluate(w, train_datos, train_clases)
            #funcion_mejora = 0.5 * tasa_clase + 0.5 * tasa_reduccion
            
            #si mejora almacenamos esos valores
            if mejor_valor_w < funcion_mejora:

                mejor_w = w
                mejor_valor_w = funcion_mejora
                mejor_tasa_clase = tasa_clase
                mejor_tasa_reduccion = tasa_reduccion
                sin_mejora = 0
                
            #si no mejora se descarta y contamos unos sin mejora
            else:
                w[pos_w] = w_anterior
                sin_mejora += 1
         
             
       
        #obtenemos la siguiente posicion a mutar
        pos_w = (pos_w + 1) % num_valores
    
    #preparar los vectores a devolver    
    tiempo = time() - start_time 
    datos_algoritmo = np.zeros(4)
    """datos_algoritmo[0] = mejor_tasa_clase
    datos_algoritmo[1] = mejor_tasa_reduccion
    datos_algoritmo[2] = mejor_valor_w
    datos_algoritmo[3] = tiempo"""
    tasa_clase, tasa_reduccion, funcion_mejora = evaluate(mejor_w, test_datos, test_clases)
    datos_algoritmo[0] = tasa_clase
    datos_algoritmo[1] = tasa_reduccion
    datos_algoritmo[2] = funcion_mejora
    datos_algoritmo[3] = tiempo

    return datos_algoritmo

#funcion para imprimir los resultados
def dibujarTabla(datos):
    for i in range(np.size(datos, 0)-1):
        datos[-1] = datos[-1] + datos[i]
    
    datos[-1] =  datos[-1] / num_particiones
    index = np.array(["1", "2", "3", "4", "5", "Media"])

    
   
    columns = np.array(["Tasa Clase", "Tasa Reduccion", "Función Objetivo", "Tiempo" ])
    
    
    
    
    tabla = pd.DataFrame(datos, index ,columns)  
    
    print(tabla)

def inicializarPoblacion(num_individuos,num_genes):
    #poblacion = np.zeros((num_individuos,num_genes))
    #for i in range(num_individuos):
    return np.random.uniform(0,1,(num_individuos, num_genes))
    
def torneoBinario(poblacion,valores,num_torneos):
    num_genes = np.size(poblacion,1)
    num_individuos = np.size(poblacion,0)
    nueva_poblacion = np.empty((num_torneos, num_genes))
    nuevos_valores = np.empty(num_torneos)
    
    for i in range(num_torneos):
        pos = random.sample(range(num_individuos), 2)
        if(valores[pos[0]] > valores[pos[1]]):
            nueva_poblacion[i] = np.copy(poblacion[pos[0]])
            nuevos_valores[i] = valores[pos[0]]
        else:
            nueva_poblacion[i] = np.copy(poblacion[pos[1]])
            nuevos_valores[i] = valores[pos[1]]
    
    return nueva_poblacion,nuevos_valores
    
    

def evaluarPoblacion(X, y, poblacion):
    num_individuos = np.size(poblacion,0)
    valores = np.zeros(num_individuos)
    for i in range(num_individuos):
        tasa_clase, tasa_reduccion, funcion_objetivo = evaluate(poblacion[i], X, y)
        valores[i] = funcion_objetivo

    return valores
        
    
def cruceBLX(cromosoma1,cromosoma2,alfa):
    num_genes = cromosoma1.size  
    hijo1 = np.empty(num_genes)
    hijo2 = np.empty(num_genes)
    for j in range(num_genes): 
        cmin = min(cromosoma1[j],cromosoma2[j])
        cmax = max(cromosoma1[j],cromosoma2[j])
        I = cmax - cmin
        h = np.random.uniform(cmin-I*alfa,cmax+I*alfa,2)
            
        if h[0] < 0:
            h[0] = 0
        if h[1] < 0:
            h[0] = 0
        if h[0] > 1:
            h[0] = 1
        if h[1] > 1:
            h[1] = 1
            
                
        hijo1[j] = h[0]
        hijo2[j] = h[1]
    
    return hijo1, hijo2

def cruceAritmetico(cromosoma1,cromosoma2):
    ponderado = np.random.uniform(0,1,1)
    hijo1 = (ponderado * cromosoma1) + ((1-ponderado) * cromosoma2)
    hijo2 = ((1-ponderado) * cromosoma1) + ( ponderado * cromosoma2)
    
    return hijo1, hijo2
            
def mutarGen(pesos, posicion):
    mutacion = np.random.normal(0.0, sigma, None)
    pesos[posicion] += mutacion
    
    return pesos
        
def obtenerPosMejorSolucion(valores):
    mejor_pos = 0
    
    for i in range(1,valores.size):
        if valores[i] > mejor_pos:
            mejor_pos = i

    return mejor_pos

def obtenerPosPeorSolucion(valores):
    peor_pos = 0
    
    for i in range(1,valores.size):
        if valores[i] < peor_pos:
            peor_pos = i

    return peor_pos
    

def AGG_BLX(training, test):
    train_datos = training[:, 0:-1]
    train_clases = np.array(training[:, -1], int)
    test_datos = test[:, 0:-1]
    test_clases = np.array(test[:, -1], int)
    
    num_pesos = np.size(train_datos,1)
    
    num_cruces = int(porcentaje_cruce * (tam_poblacion/2))
    num_mutaciones = int(porcentaje_mutacion * (tam_poblacion*num_pesos))
    
    start_time = time()
    
    poblacion = inicializarPoblacion(tam_poblacion,num_pesos)
    eval_poblacion = evaluarPoblacion(train_datos, train_clases, poblacion);
    t = tam_poblacion
    mejor_actual = obtenerPosMejorSolucion(eval_poblacion);
    w_mejor = poblacion[mejor_actual]
    w_mejor_valor = eval_poblacion[mejor_actual]
    while t < num_evaluaciones :
        
        poblacion, eval_poblacion = torneoBinario(poblacion, eval_poblacion, tam_poblacion)
        
        for i in range(num_cruces):       
            poblacion[i*2], poblacion[i*2 + 1] = cruceBLX(poblacion[i*2], poblacion[i*2 + 1], sigma)
            
            tasa_clase, tasa_reduccion, funcion_objetivo = evaluate(poblacion[i*2], train_datos, train_clases)
            eval_poblacion[i*2] = funcion_objetivo
            
            tasa_clase, tasa_reduccion, funcion_objetivo = evaluate(poblacion[i*2 + 1], train_datos, train_clases)
            eval_poblacion[i*2 + 1] = funcion_objetivo
            
            t += 2
            
        for i in range(num_mutaciones):
            pos_individuo = np.random.randint(0,tam_poblacion)
            pos_gen = np.random.randint(0,num_pesos)
            poblacion[pos_individuo] = mutarGen(poblacion[pos_individuo], pos_gen)
            
            tasa_clase, tasa_reduccion, funcion_objetivo = evaluate(poblacion[pos_individuo], train_datos, train_clases)
            eval_poblacion[pos_individuo] = funcion_objetivo
            t += 1
        
        mejor_actual = obtenerPosMejorSolucion(eval_poblacion)
        
        if(w_mejor_valor > eval_poblacion[mejor_actual]):
            peor_individuo = obtenerPosPeorSolucion(eval_poblacion)
            poblacion[peor_individuo] = w_mejor
            eval_poblacion[peor_individuo] = w_mejor_valor

        else:
            w_mejor = poblacion[mejor_actual]
            w_mejor_valor = eval_poblacion[mejor_actual]
            
            
            
        
    
    tasa_clase, tasa_reduccion, funcion_objetivo = evaluate(w_mejor, test_datos, test_clases)
    tiempo = time() - start_time 
    
    
    datos_algoritmo = np.zeros(4)
    datos_algoritmo[0] = tasa_clase
    datos_algoritmo[1] = tasa_reduccion
    datos_algoritmo[2] = funcion_objetivo
    datos_algoritmo[3] = tiempo
            
    return datos_algoritmo
        
def AGG_Aritmetico(training, test):
    train_datos = training[:, 0:-1]
    train_clases = np.array(training[:, -1], int)
    test_datos = test[:, 0:-1]
    test_clases = np.array(test[:, -1], int)
    
    num_pesos = np.size(train_datos,1)
    
    num_cruces = int(porcentaje_cruce * (tam_poblacion/2))
    num_mutaciones = int(porcentaje_mutacion * (tam_poblacion*num_pesos))
    
    start_time = time()
    
    poblacion = inicializarPoblacion(tam_poblacion,num_pesos)
    eval_poblacion = evaluarPoblacion(train_datos, train_clases, poblacion);
    t = tam_poblacion
    mejor_actual = obtenerPosMejorSolucion(eval_poblacion);
    w_mejor = poblacion[mejor_actual]
    w_mejor_valor = eval_poblacion[mejor_actual]
    while t < num_evaluaciones :
        
        poblacion, eval_poblacion = torneoBinario(poblacion, eval_poblacion, tam_poblacion)
        
        for i in range(num_cruces):       
            poblacion[i*2], poblacion[i*2 + 1] = cruceAritmetico(poblacion[i*2], poblacion[i*2 + 1])
            
            tasa_clase, tasa_reduccion, funcion_objetivo = evaluate(poblacion[i*2], train_datos, train_clases)
            eval_poblacion[i*2] = funcion_objetivo
            
            tasa_clase, tasa_reduccion, funcion_objetivo = evaluate(poblacion[i*2 + 1], train_datos, train_clases)
            eval_poblacion[i*2 + 1] = funcion_objetivo
            
            t += 2
            
        for i in range(num_mutaciones):
            pos_individuo = np.random.randint(0,tam_poblacion)
            pos_gen = np.random.randint(0,num_pesos)
            poblacion[pos_individuo] = mutarGen(poblacion[pos_individuo], pos_gen)
            
            tasa_clase, tasa_reduccion, funcion_objetivo = evaluate(poblacion[pos_individuo], train_datos, train_clases)
            eval_poblacion[pos_individuo] = funcion_objetivo
            t += 1
        
        mejor_actual = obtenerPosMejorSolucion(eval_poblacion)
        
        if(w_mejor_valor > eval_poblacion[mejor_actual]):
            peor_individuo = obtenerPosPeorSolucion(eval_poblacion)
            poblacion[peor_individuo] = w_mejor
            eval_poblacion[peor_individuo] = w_mejor_valor

        else:
            w_mejor = poblacion[mejor_actual]
            w_mejor_valor = eval_poblacion[mejor_actual]
            
            
            
        
    
    tasa_clase, tasa_reduccion, funcion_objetivo = evaluate(w_mejor, test_datos, test_clases)
    tiempo = time() - start_time 
    
    
    datos_algoritmo = np.zeros(4)
    datos_algoritmo[0] = tasa_clase
    datos_algoritmo[1] = tasa_reduccion
    datos_algoritmo[2] = funcion_objetivo
    datos_algoritmo[3] = tiempo
            
    return datos_algoritmo
        
def AGE_BLX(training, test):
    train_datos = training[:, 0:-1]
    train_clases = np.array(training[:, -1], int)
    test_datos = test[:, 0:-1]
    test_clases = np.array(test[:, -1], int)
    
    num_pesos = np.size(train_datos,1)
    
    num_mutaciones = int(1/(porcentaje_mutacion * (tam_poblacion*num_pesos)))
    num_iteraciones =  0
    
    start_time = time()
    
    poblacion = inicializarPoblacion(tam_poblacion,num_pesos)
    eval_poblacion = evaluarPoblacion(train_datos, train_clases, poblacion);
    t = tam_poblacion
    mejor_actual = obtenerPosMejorSolucion(eval_poblacion);
    w_mejor = poblacion[mejor_actual]
    w_mejor_valor = eval_poblacion[mejor_actual]
    
    while t < num_evaluaciones :
        
        nueva_poblacion, valor_poblacion = torneoBinario(poblacion, eval_poblacion, tam_poblacion)
        
             
        nueva_poblacion[0], nueva_poblacion[1] = cruceBLX(nueva_poblacion[0], nueva_poblacion[1], sigma)
            
        tasa_clase, tasa_reduccion, funcion_objetivo = evaluate(nueva_poblacion[0], train_datos, train_clases)
        valor_poblacion[0] = funcion_objetivo
        
        tasa_clase, tasa_reduccion, funcion_objetivo = evaluate(nueva_poblacion[1], train_datos, train_clases)
        valor_poblacion[1] = funcion_objetivo
            
        t += 2
        
        
            
        for i in range(num_mutaciones):
            pos_individuo = np.random.randint(0,tam_poblacion)
            pos_gen = np.random.randint(0,num_pesos)
            poblacion[pos_individuo] = mutarGen(poblacion[pos_individuo], pos_gen)
            
            tasa_clase, tasa_reduccion, funcion_objetivo = evaluate(poblacion[pos_individuo], train_datos, train_clases)
            eval_poblacion[pos_individuo] = funcion_objetivo
            t += 1
        
        mejor_actual = obtenerPosMejorSolucion(eval_poblacion)
        
        if(w_mejor_valor > eval_poblacion[mejor_actual]):
            peor_individuo = obtenerPosPeorSolucion(eval_poblacion)
            poblacion[peor_individuo] = w_mejor
            eval_poblacion[peor_individuo] = w_mejor_valor

        else:
            w_mejor = poblacion[mejor_actual]
            w_mejor_valor = eval_poblacion[mejor_actual]
            
            
            
        
    
    tasa_clase, tasa_reduccion, funcion_objetivo = evaluate(w_mejor, test_datos, test_clases)
    tiempo = time() - start_time 
    
    
    datos_algoritmo = np.zeros(4)
    datos_algoritmo[0] = tasa_clase
    datos_algoritmo[1] = tasa_reduccion
    datos_algoritmo[2] = funcion_objetivo
    datos_algoritmo[3] = tiempo
            
    return datos_algoritmo


def AGE_Aritmetico(training, test):
    train_datos = training[:, 0:-1]
    train_clases = np.array(training[:, -1], int)
    test_datos = test[:, 0:-1]
    test_clases = np.array(test[:, -1], int)
    
    num_pesos = np.size(train_datos,1)
    
    num_mutaciones = int(porcentaje_mutacion * (tam_poblacion*num_pesos))
    
    start_time = time()
    
    poblacion = inicializarPoblacion(tam_poblacion,num_pesos)
    eval_poblacion = evaluarPoblacion(train_datos, train_clases, poblacion);
    t = tam_poblacion
    mejor_solucion = obtenerPosMejorSolucion(eval_poblacion);
    while t < num_evaluaciones :
        
        
        nueva_poblacion, valores_nuevos = torneoBinario(poblacion, eval_poblacion, 2)
        
          
        nueva_poblacion[0], nueva_poblacion[1] = cruceAritmetico(nueva_poblacion[0], nueva_poblacion[1])
            
        tasa_clase, tasa_reduccion, funcion_objetivo = evaluate(nueva_poblacion[0], train_datos, train_clases)
        valores_nuevos[0] = funcion_objetivo
            
        tasa_clase, tasa_reduccion, funcion_objetivo = evaluate(nueva_poblacion[1], train_datos, train_clases)
        valores_nuevos[1] = funcion_objetivo
            
        t += 2
        
        
            
        for i in range(num_mutaciones):
            pos_individuo = np.random.randint(0,tam_poblacion)
            pos_gen = np.random.randint(0,num_pesos)
            nueva_poblacion[pos_individuo] = mutarGen(nueva_poblacion[pos_individuo], pos_gen)
            
            tasa_clase, tasa_reduccion, funcion_objetivo = evaluate(nueva_poblacion[pos_individuo], train_datos, train_clases)
            valores_nuevos[pos_individuo] = funcion_objetivo
            t += 1
        
        mejor_actual = obtenerPosMejorSolucion(valores_nuevos)
        
        if(valores_nuevos[mejor_actual] < mejor_solucion):
            peor_individuo = obtenerPosPeorSolucion(valores_nuevos)
            nueva_poblacion[peor_individuo] = poblacion[mejor_solucion]
            valores_nuevos[peor_individuo] = eval_poblacion[mejor_solucion]
            mejor_solucion = peor_individuo
        else:
            mejor_solucion = mejor_actual
            
            
        poblacion = nueva_poblacion
        eval_poblacion = valores_nuevos
    
    tasa_clase, tasa_reduccion, funcion_objetivo = evaluate(poblacion[mejor_solucion], test_datos, test_clases)
    tiempo = time() - start_time 
    
    
    datos_algoritmo = np.zeros(4)
    datos_algoritmo[0] = tasa_clase
    datos_algoritmo[1] = tasa_reduccion
    datos_algoritmo[2] = funcion_objetivo
    datos_algoritmo[3] = tiempo
            
    return datos_algoritmo   
    
    
   
    
    
def main():
    #bucle que me generara los datos con cada uno de los fichero spara 
    for i in range(3):
    

        
        #Creo la matriz donde alamacenare los valores  cada algoritmo
        """
        datos_RELIEF = np.zeros((6,4), np.float)
        datos_BL = np.zeros((6,4))
        datos_1NN = np.zeros((6,4))
        """
        datos_AGGBLX = np.zeros((6,4))
        datos_AGGAritmetico = np.zeros((6,4))
        print("------------------------")
        
        #Cargo uno de los ficheros
        if i == 0:
            data = arff.loadarff(colopscopy)
            print("MEDIDAS COLOPSCOPY")
            
        elif i == 1: 
            data = arff.loadarff(ionosphere)
            print("MEDIDAS IONOSPHERE")
        else:
            data = arff.loadarff(texture)
            print("MEDIDAS TEXTURE")
        
        #separo los datos de las etiquetas de clases.
        df = pd.DataFrame(data[0])
        clase = df.values[: , -1]      
        clase_test =  np.array(df.values[: , -1], int)
        clase = clase.reshape((1,clase.size))
        datos = df.set_index("class", drop = True)
    
        
        #normalizo los datos
        min_max_scaler = preprocessing.MinMaxScaler()
        datos = min_max_scaler.fit_transform(datos)
        
        #unimos las clases y los datos ya normalizados
        matriz_final = np.concatenate((datos,clase.T),axis=1)
        #np.random.shuffle(matriz_final)

        datos = matriz_final[:, 0:-1]
        clase = matriz_final[: , -1]
        
        #print(clase_test)
        
        #indicla la posicion en la que se almacenaran los datos del algoritmo
        i_particion = 0

        #divido los datos de respetando los portcentajes en 5 particones
        skf = StratifiedKFold(n_splits=num_particiones, shuffle=True, random_state=semilla)
        for train, test in skf.split(datos, clase_test):  
            #nos quedamos con los datos de l aparticion de train y de test
            train_datos = matriz_final[train,:]
            test_datos = matriz_final[test,:]
            
            datos_AGGAritmetico[i_particion] = AGG_Aritmetico(train_datos,test_datos)
            print(datos_AGGAritmetico[i_particion])
            #datos_AGGBLX[i_particion] = AGG_BLX(train_datos,test_datos)
            #print(datos_AGGBLX[i_particion]);
            """
            Ejecutamos y guardamos los datos para cada algoritmo
            datos_RELIEF[i_particion] = RELIEF(train_datos,test_datos)

            datos_1NN[i_particion] = k_nn(train_datos, test_datos)
            
            datos_BL[i_particion] = BL(train_datos, test_datos)
            """
            
            

            
            i_particion += 1
            
        
        #mostramos los datos al terminar los tres algoritmos
        """
        print("\nDatos RELIEF\n")
        dibujarTabla(datos_RELIEF)
        
        print("\nDatos 1-NN\n")
        dibujarTabla(datos_1NN)
        
        print("\nDatos BL\n")
        dibujarTabla(datos_BL)
        """
        print("\nDatos AGGBLX\n")
        dibujarTabla(datos_AGGAritmetico)
        
    

    






if __name__== "__main__":
  main()