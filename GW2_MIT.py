
# coding: utf-8

# ## Densidad de energía de ondas gravitacionales
# 
# Este script tiene el objetivo de obtener $$ \Omega_g(k\eta) = \frac{k^5V_k^2}{12 \pi^2 a^4 H^2}$$ para dos ecuaciones de estado distintas.
# 
# 
# 
# * Utilizando el modelo de bolsa del MIT.
# 
#     Para antes de la transición de fase resolvemos:
# 
#     $$ \ddot{V_k}a^4 \left(\frac{\tilde{B}}{2}+ \frac{\tilde{c_2}}{a^4}\right) + \dot{V_k}a^3\tilde{B} + (k^2-a^2\tilde{B})V_k = 0$$
# 
#     y para después de la transición de fase (radiación):
# 
#     $$ \ddot{V_k}+ \frac{k^2}{\tilde{c_1}}V_k=0 $$
#     
#     
# 
# * Utilizando la EoS de Nambu-JL.
# 
#     Debemos resolver:
# 
#     $$\ddot{V}_k a'^2 + \dot{V}_k a'' + \left(k^2-\frac{a''}{a}\right)V_k = 0$$
# 
# 

# In[2]:


import numpy as np
import math
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import sys


# Para resolver utilizamos un cambio de variables de la siguiente forma: 
# 
# $$ \dot{v} = w$$
# 
# 
# $$ \dot{w} = \frac{- w g(a)-v h(a)}{f(a)} $$
# 
# y resolvemos como un sistema de ecuaciones diferenciales a primer orden. $a$ entra como un array.

# In[3]:


def solnum1(H, a, k, Bb, c2b):
    
    '''
    Ecuacion diferencial para la epoca ANTERIOR a la transicion de fase.
    
    H es un array de dos valores que serán las condiciones iniciales
    a entra como un array (es la variable independiente)
    k es el modo, sera un valor fijo
    Bb se relaciona con la constante de la bolsa
    c2b constante que viene de resolver las ecuaciones de friedmann
    '''
   
    v = H[0]
    w = H[1]
    
    f = (a**4) * ( Bb/2 + c2b/(a**4) ) 
    g = (a**3) * Bb
    h = (k**2) - (a**2) * Bb

    dvdA = w
    dwdA = ( - w*g - v*h )/f

    return [dvdA, dwdA]


# In[4]:


def solnum2(H, a, k, c1b):
    
    '''
    Ecuacion diferencial para la epoca POSTERIOR a la transicion de fase,
    es decir: RADIACION.
    
    H es un array de dos valores que serán las condiciones iniciales
    a entra como un array (es la variable independiente)
    k es el modo, sera un valor fijo 
    c2b constante que viene de resolver las ecuaciones de friedmann
    '''
   
    v = H[0]
    w = H[1]
    
    dvdA = w
    dwdA = - (k**2/c1b)*v

    return [dvdA, dwdA]


# In[5]:


def entropia(T,g):
    
    '''
    Conservacion de la entropia.
    
    A partir de los grados de libertad (g) y una temperatura (T) 
    dada se puede calcular el factor de escala (a) en ese momento,
    conociendo los valores de la actualidad.
    '''
    
    a0 = 1
    T0 = 2.35/1.1605e13
    g0 = 3.95
    
    return ( ((a0*T0)**3 * g0) / (T**3 * g) )**(1/3)
    


# In[6]:


def kModo(a,T,T_tf,g,B):
    
    '''
    Fijar el modo k.
    
    Se calcula el modo a partir del factor de escala, los grados de 
    libertad y la temperatura, que es utilizada para saber si se 
    esta en la transicion de fase o en radiacion, comparando con la
    temperatura correspondiente a la transicion de fase, T_tf. 
    Segun la etapa, la densidad sera distinta.
    '''
        
    G = 6.71e-39 #GeV**-2
    
    if T>=T_tf:
        rho = (np.pi**2/30)*g*(T**4) + B
    else:
        rho = (np.pi**2/30)*g*(T**4)
    
    
    return a*math.sqrt( (8*np.pi*G*rho) / 3 )


# In[7]:


def densidadGW(Vk,k,a,H):
    
    '''
    Densidad de energia de ondas gravitacionales.
    
    Para un valor de k, y el resultado de resolver las ecuaciones diferenciales
    Vk, obtengo un punto de la grafica.
    '''             
    
    return (k**5 * Vk**2)/(12*np.pi**2*a**4*H**2)
    


# In[8]:


################################################################################
# Definimos el array de temperatura y calculamos constantes iniciales
################################################################################

T_ini = 1
T_fin = 0.011
temperatura = np.linspace(T_ini,T_fin,1000)

# Temperatura de la transicion de fase 180 MeV y posterior 179.5 MeV 

T_tf = 0.180
T_pos = 0.1795

# Calculamos la constante de la bolsa

G = 6.71e-39                                        # GeV**-2
g1 = 51.25                                          # Antes de la transicion
g2 = 17.25                                          # Despues de la transicion hasta ~ 1 MeV

B = (np.pi**2/90)*(g1-g2)*T_tf**4  
Bb = (16*np.pi*G*B)/3


# Calculamos unas constantes inciciales

rho_in = (np.pi**2/30)*g1*(T_ini**4) + B            # Densidad inicial
rho_pos = (np.pi**2/30)*g2*(T_pos**4)               # Densidad posterior a la transicion de fase
rho_fi = (np.pi**2/30)*g2*(T_fin**4)                # Densidad final (en igualdad)

c1 = (rho_pos)*(entropia(T_pos,g2)**4)
c1b = (8*np.pi*G*c1)/3
c2 = (rho_in-B)*(entropia(temperatura[0],g1)**4)
c2b = (8*np.pi*G*c2)/3

Mp = 1/math.sqrt(8*np.pi*G)                         # Masa reducida de Planck
HI = (10**(-6) * Mp) / math.sqrt(3)                 # Parámetro de Hubble en inflacion
H = (rho_fi)/(math.sqrt(3)*Mp)                      # Parámetro de Hubble en igualdad


# In[9]:


################################################################################
#Variamos la temperatura
################################################################################

densidadenergia_final1 = []
densidadenergia_final2 = []
frecuencias1 = []
frecuencias2 = []
n = 1000
# Calculo el factor de escala del momento final con g2 y el de transicion de fase con g1
akfinal = entropia(T_fin,g2)
a_tf = entropia(T_tf,g1)
jjj=0
numsol=np.zeros( (len(temperatura), n) )
for i in temperatura:
    
    if i >= T_tf:
        
        # Como Tk > T_tf, voy a utilizar g1
        g = g1
        
        # Calculamos el factor de escala
        ak = entropia(i,g)

        # Calculamos el modo que entra al horizonte en este momento
        kk = kModo(ak,i,T_tf,g,B)

        # Armamos arrays de a para resolver las ecuaciones diferenciales
        a_r1k = np.linspace(ak, a_tf, n)
        a_r2k = np.linspace(a_tf, akfinal, n)

        # Fijamos las condiciones iniciales
        V1k = (ak*HI)/math.sqrt(2*kk**3)
        V2k = HI/math.sqrt(2*kk**3)
        X0k = [V1k,V2k]

        # Resolvemos la primera ecuacion
        numerical_sol1k = odeint(solnum1, X0k, a_r1k, args=(kk, Bb, c2b))
        
        # Condiciones iniciales de la proxima integracion
        V3k = numerical_sol1k[-1,0]
        V4k = numerical_sol1k[-1,1]
        Y0k = [V3k,V4k]

        # Resolvemos la segunda ecuacion
        numerical_sol2k = odeint(solnum2, Y0k, a_r2k, args=(kk, c1b))
        numsol[jjj] = numerical_sol2k[:,0]
        jjj +=1

        # Calculamos la densidad de energia y la frecuencia
        dek = densidadGW(numerical_sol2k[-1,0],kk,akfinal,H)
        #fk = (1.65 * 2*np.pi * 1e-7 * i)* (g/100)**(1/6)
        fk = 3.53 * (i*1000/180) * (g2 / 17.25)**(1/6) * 1e-9
        
        densidadenergia_final1.append(dek)
        frecuencias1.append(fk)
        
    else:
        
        # Como Tk < T_tf, voy a utilizar g2
        g = g2
      
        # Calculamos el factor de escala
        ak = entropia(i,g)

        # Calculamos el modo que entra al horizonte en este momento
        kk = kModo(ak,i,T_tf,g,B)

        # Armamos arrays de a para resolver las ecuaciones diferenciales
        a_r2k = np.linspace(ak, akfinal, n)

        # Fijamos las condiciones iniciales
        V1k = (ak*HI)/math.sqrt(2*kk**3)
        V2k = HI/math.sqrt(2*kk**3)
        X0k = [V1k,V2k]

        # Resolvemos la segunda ecuacion
        numerical_sol2k = odeint(solnum2, X0k, a_r2k, args=(kk, c1b))

        # Calculamos la densidad de energia y la frecuencia
        dek = densidadGW(numerical_sol2k[-1,0],kk,akfinal,H)
        #fk = (1.65 * 2*np.pi * 1e-7 * i)* (g/100)**(1/6)
        fk = 3.53 * (i*1000/180) * (g2 / 17.25)**(1/6) * 1e-9
        
        densidadenergia_final2.append(dek)
        frecuencias2.append(fk)
        


# In[10]:


################################################################################
#Realizamos gráficos
################################################################################

plt.figure(figsize=(15,6))

densidadenergia_final = densidadenergia_final1 + densidadenergia_final2
frecuencias = frecuencias1 + frecuencias2

f_tf = 3.53 * (T_tf*1000/180) * (g2 / 17.25)**(1/6) * 1e-9

#plt.subplot(212)
plt.plot(frecuencias[:], np.array(densidadenergia_final)/np.max(np.array(densidadenergia_final)), 
         '-o', color = 'b', alpha=0.6)
plt.axvline(x=f_tf, color = 'm', ls='--', lw=3, label='Transición de fase \n MIT')
plt.xlabel('f [Hz]', fontsize=20)
plt.ylabel('$\Omega_g(f)$', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xscale('log')
plt.legend(loc = 'best', fontsize=18)
#plt.savefig('/home/paulalopez/Escritorio/Figuras/Transicion_MIT.png')

mm = [frecuencias,np.array(densidadenergia_final)/np.max(np.array(densidadenergia_final))]

np.savetxt('tabla_mit.txt', np.array(mm).T)

#plt.grid()


# In[11]:


'''

Vemos como evolucionan las soluciones numericas en funcion del
factor de escala, para 3 k (modos) distintos.

2 corresponden a modos que ingresaron antes de la transicion
de fase (k > k*)
1 corresponde a un modo que ingreso despues de la transicion
de fase (k < k*)

La transicion de fase ocurrio a T_tf = 0.18 GeV

'''
# Inicializo arrays

numerical_sol12 = []
numerical_sol22 = []

numerical_sol13 = []
numerical_sol23 = []

numerical_sol24 = []

# Pongo condiciones para temperatura inicial y final y los factores de escala correspondientes

T_ini = 5
T_fin = 0.01

ainicial = entropia(T_ini,g1)
akfinal = entropia(T_fin,g2)

# Modos que ingresaron antes de la transicion de fase: T1=5 GeV y T2=0.3 GeV 
T2 = 5
a2 = entropia(T2,g1)
k2 = kModo(a2,T2,T_tf,g1,B)
a_2 = np.linspace(a2,a_tf,1000)
a_22 = np.linspace(a_tf, akfinal, 1000)
# Fijamos las condiciones iniciales
V1k2 = (a2*HI)/math.sqrt(2*k2**3)
V2k2 = HI/math.sqrt(2*k2**3)
X0k2 = [V1k2,V2k2]
# Resolvemos la primera ecuacion
numerical_sol12 = odeint(solnum1, X0k2, a_2, args=(k2, Bb, c2b))
# Condiciones iniciales de la proxima integracion
V3k2 = numerical_sol12[-1,0]
V4k2 = numerical_sol12[-1,1]
Y0k = [V3k2,V4k2]
# Resolvemos la segunda ecuacion
numerical_sol22 = odeint(solnum2, Y0k, a_22, args=(k2, c1b))[:,0]

T3 = 0.3
a3 = entropia(T3,g1)
k3 = kModo(a3,T3,T_tf,g1,B)
a_3 = np.linspace(a3,a_tf,1000)
a_32 = np.linspace(a_tf, akfinal, 1000)
# Fijamos las condiciones iniciales
V1k3 = (a3*HI)/math.sqrt(2*k3**3)
V2k3 = HI/math.sqrt(2*k3**3)
X0k3 = [V1k3,V2k3]
# Resolvemos la primera ecuacion
numerical_sol13 = odeint(solnum1, X0k3, a_3, args=(k3, Bb, c2b))
# Condiciones iniciales de la proxima integracion
V3k3 = numerical_sol13[-1,0]
V4k3 = numerical_sol13[-1,1]
Y0k3 = [V3k3,V4k3]
# Resolvemos la segunda ecuacion
numerical_sol23 = odeint(solnum2, Y0k3, a_32, args=(k3, c1b))[:,0]

T4 = 0.1
a4 = entropia(T4,g2)
k4 = kModo(a4,T4,T_tf,g2,B)
a_4 = np.linspace(a4, akfinal, 1000)
# Fijamos las condiciones iniciales
V1k4 = (a4*HI)/math.sqrt(2*k4**3)
V2k4 = HI/math.sqrt(2*k4**3)
X0k = [V1k,V2k]
# Resolvemos la segunda ecuacion
numerical_sol24 = odeint(solnum2, X0k, a_4, args=(k4, c1b))[:,0]


# In[12]:


################################################################################
#Realizamos gráficos
################################################################################

plt.figure(figsize=(16,8))
plt.plot(a_2[:]/a_2[0], ((numerical_sol12[:,0]/V1k2)/(a_2/a_2[0])), '-', color = 'g',label='k={:3.1e}'.format(k2))
plt.plot(a_22[:]/a_2[0], ((numerical_sol22/V1k2)/(a_22/a_2[0])), '-', color = 'g')
plt.plot(a_3[:]/a_2[0], ((numerical_sol13[:,0]/V1k3)/(a_3/a_3[0])), '-', color = 'r',label='k={:3.1e}'.format(k3))
plt.plot(a_32[:]/a_2[0], ((numerical_sol23/V1k3)/(a_32/a_3[0])), '-', color = 'r')
plt.plot(a_4[:]/a_2[0], (((numerical_sol24)/V1k)/(a_4/a_4[0])), '-', color = 'b',label='k={:3.1e}'.format(k4))
plt.axvline(x=a2/a_2[0], color = 'g', ls='--')
plt.axvline(x=a4/a_2[0], color = 'b', ls='--')#, lw=2)
plt.axvline(x=a3/a_2[0], color = 'r', ls='--')
plt.axvline(x=a_tf/a_2[0], color = 'k', ls='--', label='Transición de fase')
plt.xlabel('$a(T)/a(T_{in})$', fontsize=16)
plt.ylabel('$h/h_{prim}$', fontsize=16)
plt.legend(loc = 'best', fontsize=16)
#plt.savefig('/home/pau/Escritorio/Figuras/Modos_MIT.png')
#plt.grid()


# In[17]:


################################################################################
#Realizamos gráficos
################################################################################

plt.figure(figsize=(16,8))
plt.plot(a_2[:]/a_2[0], ((numerical_sol12[:,0]/V1k2)/(a_2/a_2[0]))**2, '-',
         color = 'g',label='k={:3.1e}'.format(k2), lw=4)

plt.plot(a_22[:]/a_2[0], ((numerical_sol22/V1k2)/(a_22/a_2[0]))**2, '-', color = 'g')
plt.plot(a_3[:]/a_2[0], ((numerical_sol13[:,0]/V1k3)/(a_3/a_3[0]))**2, '-', color = 'r',
         label='k={:3.1e}'.format(k3), lw=4)

plt.plot(a_32[:]/a_2[0], ((numerical_sol23/V1k3)/(a_32/a_3[0]))**2, '-', color = 'r', lw=4)
plt.plot(a_4[:]/a_2[0], (((numerical_sol24)/V1k)/(a_4/a_4[0]))**2, '-', color = '#1167ca',
         label='k={:3.1e}'.format(k4), lw=4)

plt.axvline(x=a2/a_2[0], color = 'g', ls='--', lw =3)
plt.axvline(x=a4/a_2[0], color = '#1167ca', ls='--', lw=3)
plt.axvline(x=a3/a_2[0], color = 'r', ls='--', lw=3)
plt.axvline(x=a_tf/a_2[0], color = 'k', ls='--', label='Transición de fase', lw=2)
plt.xlabel('$a(T)/a(T_{in})$', fontsize=20)
plt.ylabel('$(h/h_{prim})^2$', fontsize=20)
plt.legend(loc = 'best', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlim(0,400)
#y, xmin, xmax
#plt.savefig('/home/paulalopez/Escritorio/Figuras/Modos_MIT.png')
#plt.grid()


# In[14]:


presion = (np.pi**2*g1*temperatura**4)/90 - B
plt.plot(temperatura, presion, 'r')
#plt.hlines((presion/temperatura**4)[0], 0, temperatura[0], color='r')


# In[15]:


################################################################################
#Realizamos gráficos
################################################################################


#plt.figure(figsize=(16,8))
#plt.plot(a_1[:], numerical_sol11[:]/2.40183853e+60, '--', color = 'r',label='k={}'.format(k1))
#plt.plot(a_12[:], numerical_sol21[:]/2.40183853e+60, '--', color = 'r')
#plt.plot(a_2[:], numerical_sol12[:]/2.40183853e+60, '--', color = 'g',label='k={}'.format(k2))
#plt.plot(a_22[:], numerical_sol22[:]/2.40183853e+60, '--', color = 'g')
#plt.plot(a_3[:], numerical_sol13[:]/2.40183853e+60, '--', color = 'm',label='k={}'.format(k3))
#plt.plot(a_32[:], numerical_sol23[:]/2.40183853e+60, '--', color = 'm')
#plt.plot(a_4[:], numerical_sol24 /2.40183853e+60, '--', color = 'b',label='k={}'.format(k4))

#plt.axvline(x=a1, color = 'r')
#plt.axvline(x=a2, color = 'g')
#plt.axvline(x=a3, color = 'm')
#plt.axvline(x=a4, color = 'b')
#plt.axvline(x=a_tf, color = 'y', label='Transición de fase')

#plt.xscale('log')
#plt.xlim(ainicial,akfinal)
#plt.ylim(2.4e60,2.402e60)

#plt.xlabel('$a(T)$', fontsize=16)
#plt.ylabel('$V(a)$', fontsize=16)

#plt.legend(loc = 'lower left', fontsize='medium')
#plt.grid()

#plt.savefig('/home/paulalopez/Escritorio/Numersol.png')

#mask= np.where(numerical_sol23 > 2.40175e60)
#aux = mask[0]
#print(aux)
#print(numerical_sol23[679])


# In[ ]:


'''

Vemos como evolucionan las soluciones numericas en funcion del
factor de escala, para 4 k (modos) distintos.

3 corresponden a modos que ingresaron antes de la transicion
de fase (k > k*)
1 corresponde a un modo que ingreso despues de la transicion
de fase (k < k*)

La transicion de fase ocurrio a T_tf = 0.18 GeV

'''
# Inicializo arrays

#numerical_sol11 = []
#numerical_sol12 = []
#numerical_sol14 = []

#numerical_sol21 = []
#numerical_sol22 = []
#numerical_sol23 = []
#numerical_sol24 = []

# Pongo condiciones para temperatura inicial y final y los factores de escala correspondientes

#T_ini = 1
#T_fin = 0.011

#ainicial = entropia(T_ini,g1)
#akfinal = entropia(T_fin,g2)

# Modos que ingresaron antes de la transicion de fase: T1=0.8 GeV, T2=0.5 GeV y T3=0.2 GeV

#T1 = 0.8
#a1 = entropia(T1,g1)
#k1 = kModo(a1,T1,T_tf,g1,B)
#a_1 = np.linspace(a1,a_tf,1000)
#a_12 = np.linspace(a_tf, akfinal, 1000)
# Fijamos las condiciones iniciales
#V1k = (a1*HI)/math.sqrt(2*k1**3)
#V2k = HI/math.sqrt(2*k1**3)
#X0k = [V1k,V2k]
# Resolvemos la primera ecuacion
#numerical_sol11 = odeint(solnum1, X0k, a_1, args=(k1, Bb, c2b))
# Condiciones iniciales de la proxima integracion
#V3k = numerical_sol11[-1,0]
#V4k = numerical_sol11[-1,1]
#Y0k = [V3k,V4k]
# Resolvemos la segunda ecuacion
#numerical_sol21 = odeint(solnum2, Y0k, a_12, args=(k1, c1b))



#T2 = 0.5
#a2 = entropia(T2,g1)
#k2 = kModo(a2,T2,T_tf,g1,B)
#a_2 = np.linspace(a2,a_tf,1000)
#a_22 = np.linspace(a_tf, akfinal, 1000)
# Fijamos las condiciones iniciales
#V1k = (a2*HI)/math.sqrt(2*k2**3)
#V2k = HI/math.sqrt(2*k2**3)
#X0k = [V1k,V2k]
# Resolvemos la primera ecuacion
#numerical_sol12 = odeint(solnum1, X0k, a_2, args=(k2, Bb, c2b))
# Condiciones iniciales de la proxima integracion
#V3k = numerical_sol12[-1,0]
#V4k = numerical_sol12[-1,1]
#Y0k = [V3k,V4k]
# Resolvemos la segunda ecuacion
#numerical_sol22 = odeint(solnum2, Y0k, a_22, args=(k2, c1b))


#T3 = 0.3
#a3 = entropia(T3,g1)
#k3 = kModo(a3,T3,T_tf,g1,B)
#a_3 = np.linspace(a3,a_tf,1000)
#a_32 = np.linspace(a_tf, akfinal, 1000)
# Fijamos las condiciones iniciales
#V1k = (a3*HI)/math.sqrt(2*k3**3)
#V2k = HI/math.sqrt(2*k3**3)
#X0k = [V1k,V2k]
# Resolvemos la primera ecuacion
#numerical_sol13 = odeint(solnum1, X0k, a_3, args=(k3, Bb, c2b))
# Condiciones iniciales de la proxima integracion
#V3k = numerical_sol13[-1,0]
#V4k = numerical_sol13[-1,1]
#Y0k = [V3k,V4k]
# Resolvemos la segunda ecuacion
#numerical_sol23 = odeint(solnum2, Y0k, a_32, args=(k3, c1b))


# Modo que ingreso luego de la transicion de fase: T4=0.1 GeV

#T4 = 0.1
#a4 = entropia(T4,g2)
#k4 = kModo(a4,T4,T_tf,g2,B)
#a_3 = np.linspace(a3,a_tf,1000)
#a_4 = np.linspace(a4, akfinal, 1000)
# Fijamos las condiciones iniciales
#V1k = (a4*HI)/math.sqrt(2*k4**3)
#V2k = HI/math.sqrt(2*k4**3)
#X0k = [V1k,V2k]
# Resolvemos la segunda ecuacion
#numerical_sol24 = odeint(solnum2, X0k, a_4, args=(k4, c1b))[:,0]


# In[ ]:


# Trabajo por un lado con la parte de radiacion
# Defino como arrays a la listas que contienen las densidades y frecuencias

#densidadenergia_final2 = np.array(densidadenergia_final2)
#frecuencias2 = np.array(frecuencias2)

# Defino, mediante una mascara, una tupla que contiene las ubicaciones (dentro del array definido
# arriba) de las densidades que nos interesan

#mask2= np.where(densidadenergia_final2 > 2.001e29)
#print(np.shape(mask2[0]))

# Defino arrays auxiliares para guardar los valores que me importan (Donde mask[0] contiene las 
# ubicaciones de los valores que me interesan), de la siguiente manera:

#aux2 = densidadenergia_final2[mask2[0]]
#faux2 = frecuencias2[mask2[0]]
#print(aux2)
#print(faux2)

# Finalmente grafico los puntos

#plt.plot(faux2, aux2, '-*', color = 'r')
#plt.xlabel('Frecuencias')
#plt.ylabel('$\Omega_g$')
#plt.xscale('log')
#plt.grid()



# In[ ]:


#plt.figure(figsize=(16,8))

#plt.subplot(212)
#plt.plot(frecuencias1[:], densidadenergia_final1[:], '-*', color = 'g')
#plt.xlabel('Frecuencias')
#plt.ylabel('$\Omega_g$')
#plt.xscale('log')
#plt.xlim(8.5e-7,9.5e-7)
#plt.ylim(1.4e29,1.8e29)
#plt.grid()


# In[ ]:


# Trabajo ahora con la parte de la transicion de fase
# Defino como arrays a la listas que contienen las densidades y frecuencias

#densidadenergia_final1b = np.array(densidadenergia_final1)
#frecuencias1b = np.array(frecuencias1)

# Defino, mediante una mascara, una tupla que contiene las ubicaciones (dentro del array definido
# arriba) de las densidades que nos interesan 

#indice = np.logical_and(densidadenergia_final1b > 1.4e29,1.85e29 > densidadenergia_final1b, )
#mask1= np.where(indice)
#print(np.shape(mask1[0]))
#aux1= densidadenergia_final1b[mask1[0]]
#faux1 = frecuencias1b[mask1[0]]
#print(aux1)
#print(faux1)

#indice = np.logical_and(densidadenergia_final1b > 1.45e29,1.85e29 > densidadenergia_final1b, )
#mask1= np.where(indice)
#print(np.shape(mask1[0]))
#aux1= densidadenergia_final1b[mask1[0]]
#faux1 = frecuencias1b[mask1[0]]
#print(aux1)
#print(faux1)

# Defino arrays auxiliares para guardar los valores que me importan (Donde mask[0] contiene las 
# ubicaciones de los valores que me interesan), de la siguiente manera:



# Finalmente grafico los puntos
#plt.plot(faux1, aux1, '-*', color = 'r')
#plt.xlabel('Frecuencias')
#plt.ylabel('$\Omega_g$')
#plt.xscale('log')
#plt.grid()


# In[ ]:


#plt.figure(figsize=(9,5))

#plt.plot(faux1, aux1/2.00740186e29, '-*', color = 'r')
#plt.plot(faux2, aux2/2.00740186e29, '-*', color = 'r')
#plt.xlabel('Frecuencias')
#plt.ylabel('$\Omega_g$')
#plt.xscale('log')
#plt.xlim(1e-8,1e-6)
#plt.ylim(0.6,1.1)
#plt.grid()
#plt.savefig('/home/paulalopez/Escritorio/DensidadEnergiaGW.png')
#plt.savefig('/home/paulalopez/Escritorio/DensidadEnergiaGW.pdf')


# In[ ]:


#fg = np.mean(numsol, axis=1)
#fg2 = np.mean(numsol, axis=1)
#print(len(fg))
#print()
#for i in range(200):
#    plt.plot(i,fg2[i]/1e45, 'k-o')
#plt.axvline(127)


# In[ ]:


#from scipy.optimize import curve_fit


#dens_fit = np.array(densidadenergia_final[:])

#def sinfit(k, amp,a,w):
#    return amp*np.sin(k*a+w)

#popt, pcov = curve_fit(sinfit, frecuencias[:], dens_fit)



# In[ ]:


#plt.plot(frecuencias[:], dens_fit, '-', color = 'c')
#plt.plot(frecuencias, sinfit(frecuencias, *popt))
#plt.xlabel('Frecuencias')
#plt.ylabel('$\Omega_g$')
#plt.xscale('log')
#plt.grid()


# In[ ]:


#print(entropia(0.001,g2))


# In[20]:


x1, y1 = np.loadtxt('tabla_mit.txt', unpack=True)
x2, y2 = np.loadtxt('tabla_nambu.txt', unpack=True)

plt.figure(figsize=(15,9))
plt.plot(x1,y1, color='m')
plt.plot(x2,y2, color='g')

