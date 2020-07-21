
# coding: utf-8

# ## Nambu
# 
# 

# In[1]:


import numpy as np
import math
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import sys
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# In[2]:


def solnum_total(H, a, k, *opt):
    
    '''
    Ecuacion diferencial para la epoca POSTERIOR
    a la transicion de fase.
    
    H es un array de dos valores que serán las condiciones iniciales
    a entra como un array (es la variable independiente)
    k es el modo, sera un valor fijo
    '''
    
    v = H[0]
    u = H[1]
    
    
    G = 6.71e-39 #GeV**-2    
    
    pw = opt[3:]
    
    rho = opt[0]/(a**opt[2]) + opt[1] 

    w = pw[0]+pw[1]/(pw[2]+pw[3]*a**4)+pw[4]*a+pw[5]*a**2+pw[6]*a**3
    
    f = (8*np.pi*G / 3)*(a**4) * rho
    g = (4*np.pi*G / 3)*(a**3)  * rho * ( 1 - 3*w )
    h = (k**2) - ((4*np.pi*G / 3)*(a**2)  * rho * ( 1 - 3*w ))

    dvdA = u
    dudA = ( - u*g - v*h )/f

    return [dvdA, dudA]


# In[3]:


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
    


# In[4]:


def kModo(a,rho, verbose = False):
    
    '''
    Fijar el modo k.
    
    Se calcula el modo a partir del factor de escala
    y la densidad tomada de tabla.
    '''
        
    G = 6.71e-39 #GeV**-2
    if verbose: print(rho)
    return a*math.sqrt( (8*np.pi*G*rho) / 3 )


# In[5]:


def densidadGW(Vk,k,a,H):
    
    '''
    Densidad de energia de ondas gravitacionales.
    
    Para un valor de k, y el resultado de resolver las ecuaciones diferenciales
    Vk, obtengo un punto de la grafica.
    '''             
    
    return (k**5 * Vk**2)/(12*np.pi**2*a**4*H**2)
    


# In[6]:


# Abro el archivo donde tengo la ecuacion de estado: Densidad, Presion, Temperatura

rho, P, T = np.loadtxt('EoS_NJL.dat',unpack=True)

F = rho - 3*P
mask = np.where(F>0.0)

Fnew = F[mask]
T = T[mask]
rho = rho[mask]
P= P[mask]
#plt.plot(T,F/T**4)
#print(len(F))
#print(np.mean(np.diff(P)/np.diff(rho)))
#plt.grid()


# In[24]:


################################################################################
# Definimos el array de temperatura y calculamos constantes iniciales
################################################################################

# Temperatura de la transicion de fase 155.6 MeV y posterior 156.5 MeV 

T_tf = 0.1556

# Algunos valores a considerar

G = 6.71e-39                                          # GeV**-2
g1 = 51
#g1 = 51                                               # Antes de la transicion
g2 = g1
#g2 = 21.5                                             # Despues de la transicion hasta ~ 1 MeV

Mp = 1/math.sqrt(8*np.pi*G)                           # Masa reducida de Planck
HI = (10**(-6) * Mp) / math.sqrt(3)                   # Parámetro de Hubble en inflacion
H = (rho[-1])/(math.sqrt(3)*Mp)                       # Parámetro de Hubble en igualdad


# In[25]:


B = (np.pi**2/90)*(g1-g2)*T_tf**4  
tmaska = np.where(T[T>T_tf])
tmaskd = np.where(T[T<T_tf])

rhoMITa = (8*np.pi/15)*(T[tmaska]**4) + B
PMITa = (8*np.pi/45)*(T[tmaska]**4) - B
rhoMITd = (8*np.pi/15)*(T[tmaskd]**4)
PMITd = (8*np.pi/45)*(T[tmaskd]**4)


plt.clf()
#plt.plot(T[:-1],np.diff(P)/np.diff(rho))
#plt.plot(T,P/rho)
plt.xlim(0.005,0.9)
#plt.ylim(0,7)
plt.plot(T,(rho-3*P)/T**4, 'b', lw=3, label = 'NJL')
plt.plot(T[tmaska],(rhoMITa-3*PMITa)/T[tmaska]**4, 'g', lw=3, label = 'MIT')
plt.plot(T[tmaskd],(rhoMITd-3*PMITd)/T[tmaskd]**4, 'g', lw=3)
plt.axvline(T_tf, color='k')
plt.xlabel('T[GeV]', fontsize= 20)
plt.text(0.4, 6.5,r'$\frac{\rho - 3P}{T^4}$', rotation = 0, fontsize= 20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(loc='best', fontsize = 16)
plt.grid()
#plt.plot(rho, P, 'b', lw=3, label = 'NJL')
#plt.plot(rhoMITa, PMITa, 'g', lw=3, label = 'MIT')
#plt.plot(rhoMITd, PMITd, 'g', lw=3)


# ## Realizamos ajuste de w(a). Antes y despues de la transicion y tambien en total.

# In[26]:


def w_tot(a, c0, c1, c2, c3, n1, n2, n3):
    return c0 + c1/(c2+c3*a**4) + n1*a + n2*a**2+ n3*a**3


# In[27]:


################################################################################
#Fiteo
################################################################################

xdata_ant = []
xdata_des = []
ydata_ant = []
ydata_des = []

xdata_tot = []
ydata_tot = []

rho_ant = []
rho_des = []


for j,i in enumerate(T):
    if i >= T_tf:
        xdata_ant.append(entropia(i,g1))
        ydata_ant.append(P[j]/rho[j])
        rho_ant.append(rho[j])
        
    else:
        xdata_des.append(entropia(i,g2))
        ydata_des.append(P[j]/rho[j]) 
        rho_des.append(rho[j])
        
# Primer argumento de curve_fit: la funcion modelo, a la que le quiero determinar los parametros
# y que tiene como primer argumento la variable independiente
# Segundo argumento de curve_fit: la variable independiente, donde la informacion es evaluada
# Tercer argumento de curve_fit: la data

#********************************************************************************************************

xdata_tot = xdata_ant + xdata_des
xdata_total = np.array(xdata_tot)/xdata_tot[0]
ydata_tot = np.array(P/rho)


popt_tot, pcov_tot = curve_fit(w_tot, xdata_total, ydata_tot)
print(popt_tot)

#********************************************************************************************************

#********************************************************************************************************


# ## Graficamos w(a) con la funcion fiteada, en la totalidad

# * El ajuste necesita de la data normalizada.

# In[28]:


plt.figure(figsize=(15,8))
plt.plot(xdata_tot, ydata_tot, 'o', color='#a620f3', label='Datos')
plt.plot(xdata_tot, w_tot(xdata_total, *popt_tot), color='k', label='Funcion fiteada ')
plt.legend(loc = 'best', fontsize=18)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('a', fontsize=20)
plt.ylabel('w(a)', fontsize=20)
plt.savefig('/home/pau/Escritorio/Figuras/w_a.png')
#plt.savefig('/home/paulalopez/Escritorio/Tesis_febrero/Figuras/w_a.png')


# ## Una vez que tenemos w(a) analiticamente, resolvemos la ecuacion de movimiento
# ## de rho(a).  Nuevamente separamos el analisis entre antes y despues de la transicion y en  su totalidad.

# In[29]:


def funcionrho_total (H,a, *opt):
    
    rho = H[0]

    w_t = opt[0]+opt[1]/(opt[2]-opt[3]*a**4)+opt[4]*a+opt[5]*a**2+opt[6]*a**3

    f = - (3/a)*(1 + w_t )
               
    rhoA = f*rho
    
    return [rhoA]


# In[30]:


# Resuelvo la ecuacion de movimiento de rho dada por RG
# esto me va a dar valores de densidad de energia en 
# funcion del factor de escala

densidad_tot = odeint(funcionrho_total,rho[0], xdata_tot, args=(*popt_tot,))


# ## Graficamos rho(a)

# In[31]:


################################################################################
#Realizamos gráficos
################################################################################
plt.figure(figsize=(12,6))

plt.plot(xdata_tot, densidad_tot, 'r')


# ## Ajustamos funciones a rho, para obtener la expresion analitica.

# In[32]:


def frho_tot(a, c0, c1,c2):
    return c0/a**c2 + c1


# In[33]:


poptrho_tot, pcovrho_tot = curve_fit(frho_tot, xdata_total, densidad_tot[:,0])
print(poptrho_tot)  


# In[34]:


################################################################################
#Realizamos gráficos
################################################################################

plt.figure(figsize=(12,6))

plt.xlabel('Factor de escala a', fontsize=20)
plt.ylabel('rho(a)', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.plot(xdata_total*xdata_tot[0], frho_tot(xdata_total, *poptrho_tot), color='r', lw=3, label='Funcion fiteada')
plt.legend(loc = 'best', fontsize=16)

#plt.savefig('/home/paulalopez/Escritorio/Tesis_febrero/Figuras/rho_a.png')


# ## Una vez ajustadas, escribimos explicitamente las funciones

# In[35]:


def rhoA(a, *popt):
    rho = 0

    for j, ipar in enumerate(popt):
        rho += ipar*a**j

    return rho


# In[36]:


def rhoD(a, *popt):

    rho = 0.

    for j,ipar in enumerate(popt):
        rho += ipar*a**j
    
    return rho


# In[37]:



ptest_tot = list(poptrho_tot)+list(popt_tot)


# ## Pasamos a resolver la ecuacion diferencial de Vk

# In[38]:


################################################################################
#Variamos la temperatura
################################################################################

densidadenergia_final1 = []
frecuencias1 = []
densidadenergia_final2 = []
frecuencias2 = []

# Calculo el factor de escala del momento final con g2 y el de transicion de fase con g1
akfinal = entropia(T[-1],g2)
a_tf = entropia(T_tf,g1)
n = 10000
g1 = 41.75
densidadenergia_final1_j = []
densidadenergia_final2_j = []
#for i in temperatura:
for i in T:
    
    if i >= T_tf:
        
        # Como Tk > T_tf, voy a utilizar g1
        g = g1
        
        # Calculamos el factor de escala
        ak = entropia(i,g)
        
        #Agregado por Gabi, generalizado
        rhok = frho_tot(ak, *poptrho_tot)

        # Calculamos el modo que entra al horizonte en este momento
        kk = kModo(ak,rhok)#(ak, *poptrho_ant), verbose = False)

        # Armamos arrays de a para resolver las ecuaciones diferenciales
        a_r1k = np.linspace(ak, a_tf, n)
        a_r2k = np.linspace(a_tf, akfinal, n)
        
        a_r1k = a_r1k#/a_r1k[0]
        a_r2k = a_r2k#/a_r2k[0]
        # Fijamos las condiciones iniciales
        V1k = (ak*HI)/math.sqrt(2*kk**3)
        V2k = HI/math.sqrt(2*kk**3)
        X0k = [V1k,V2k]

        # Resolvemos la primera ecuacion
        #numerical_sol1k = odeint(solnum_antes, X0k, a_r1k, args=(kk,*ptest_ant,))
        numerical_sol1k_j = odeint(solnum_total, X0k, a_r1k, args=(kk,*ptest_tot))

        # Condiciones iniciales de la proxima integracion
        V3k = numerical_sol1k_j[-1,0]
        V4k = numerical_sol1k_j[-1,1]
        Y0k = [V3k,V4k]

        # Resolvemos la segunda ecuacion
        #numerical_sol2k = odeint(solnum_despues, Y0k, a_r2k, args=(kk,*ptest_des))
        numerical_sol2k_j = odeint(solnum_total, Y0k, a_r2k, args=(kk,*ptest_tot))


        # Calculamos la densidad de energia y la frecuencia
        #dek = densidadGW(numerical_sol2k[-1,0],kk,akfinal,H)
        #dek = densidadGW(numerical_sol2k[-1,0],kk,akfinal,H)
        dek_j = densidadGW(numerical_sol2k_j[-1,0],kk,akfinal,H)
        fk = 3.53 * (i*1000/180) * (g1 / 17.25)**(1/6) * 1e-9
        
        #densidadenergia_final1.append(dek)
        densidadenergia_final1_j.append(dek_j)
        frecuencias1.append(fk)
        
    else:
        # Como Tk < T_tf, voy a utilizar g2
        g = g1
      
        # Calculamos el factor de escala
        ak = entropia(i,g)

        rhok = frho_tot(ak, *poptrho_tot)
        
        # Calculamos el modo que entra al horizonte en este momento
        #print(rhoA(ak, *popt_ant))
        #kk = kModo(ak,rhoD(ak))
        kk = kModo(ak,rhok)#D(ak, *poptrho_des), verbose = False)

        # Armamos arrays de a para resolver las ecuaciones diferenciales
        a_r2k = np.linspace(ak, akfinal, n)
        a_r2k = a_r2k#/a_r2k[0]
        
        # Fijamos las condiciones iniciales
        V1k = (ak*HI)/math.sqrt(2*kk**3)
        V2k = HI/math.sqrt(2*kk**3)
        X0k = [V1k,V2k]

        # Resolvemos la segunda ecuacion
        #numerical_sol2k = odeint(solnum_despues, X0k, a_r2k, args=(kk,*ptest_des))
        numerical_sol2k_j = odeint(solnum_total, X0k, a_r2k, args=(kk,*ptest_tot))

        # Calculamos la densidad de energia y la frecuencia
        #dek = densidadGW(numerical_sol2k[-1,0],kk,akfinal,H)
        #dek = densidadGW(numerical_sol2k[-1,0],kk,akfinal,H)
        dek_j = densidadGW(numerical_sol2k_j[-1,0],kk,akfinal,H)
        fk = 3.53 * (i*1000/180) * (g1 / 17.25)**(1/6) * 1e-9
        
        #densidadenergia_final2.append(dek)
        densidadenergia_final2_j.append(dek_j)
        frecuencias2.append(fk)
        


# In[39]:


################################################################################
#Realizamos gráficos
################################################################################
densidadenergia_final = densidadenergia_final1 + densidadenergia_final2
frecuencias = frecuencias1 + frecuencias2
densidadenergia_final_j = densidadenergia_final1_j + densidadenergia_final2_j

plt.figure(figsize=(15,6))

f_tf = 3.53 * (T_tf*1000/180) * (g1 / 17.25)**(1/6) * 1e-9

#plt.plot(frecuencias, np.array(densidadenergia_final), '-o', color = 'b')
plt.plot(frecuencias, np.array(densidadenergia_final_j)/np.max(np.array(densidadenergia_final_j)),
         '-o', color = 'b', alpha=0.6)
plt.axvline(x=f_tf, color = 'm', ls='--', lw=3, label='Transición de fase \n PNJL-nl')
plt.xlabel('f [Hz]', fontsize=20)
plt.ylabel('$\Omega_g(f)$', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xscale('log')
plt.legend(loc = 'best', fontsize=18)
##plt.savefig('/home/paulalopez/Escritorio/Tesis_febrero/Figuras/Transicion_PNJL.png')
#plt.savefig('/home/pau/Escritorio/Figuras/Transicion_PNJL.png')
#plt.savefig('/home/paulalopez/Escritorio/Tesis_febrero/Figuras/Transicion_PNJL.png')

