#!/usr/bin/env python
# coding: utf-8

from QGD_functions import*
#from itertools import permutations

n=3
disent_order=list(range(n-1,-1,-1))
dis = disent_dict(disent_order)
mode=['cz','U3']
initial='qft'

opt_type=['Sim',False]
cf_type=['Disentangle','F','merged']

#Lista de posibilidades:
#lista = PoolCycles(n)
#lista.reverse()


#opt_parameters=[eps,tol,Nmax,Nopt,Coste_exit,Coste_extend]
#Nmax: nº max de pasos en la optimizacion (en general si la optimizacion va bien, se va a extender hasta llegar al eps, ignoramos tolerancia y a las malas acaba en break si superamos las 100 vueltas)
#Nopt: si opt_type[2]==True nº max de intentos de optimizacion de la layer (nos quedamos con el mejor)
#Coste_exit: Cota de Coste mínimo que tiene que tener la función de coste para continuar con la optimización


#Default parameters:
#opt_parameters=[1e-6,1e-6,20,6,20,5]

opt_parameters=[1e-12,1e-10,35,6,20,5]

cycles=[3,1]

Max_tries=100

conditions = [n,dis,cycles,mode,initial]
p=MPD(conditions)
p0=MPDIni(conditions)

eps,tol,Nmax,Nopt,Coste_exit,Coste_extend = opt_parameters

print("-----------------------------------------------")
print("Configuracion",cycles)
print("----------------------------------------------- \n\r")

print(conditions)
print(opt_type)
print(cf_type)

Contador=0
Resultado=[100,100]
while Resultado[0]>eps:
    if Contador>Max_tries:
        break
    Result=Optimizacion(conditions,cf_type,opt_type,opt_parameters,displays=True)
    print("Resultado:",Result)
    if Result[0]<Resultado[0]:
        Resultado=Result
    Contador+=1
Param=Resultado[1]
name ="%dqbits_%s_%d_%d.txt" % (n,mode[0],cycles[0],cycles[1])
save_to_file(Param,name)
