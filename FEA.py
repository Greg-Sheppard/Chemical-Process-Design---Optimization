import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

np.set_printoptions(precision=3,  threshold=np.nan)

#Returns heat transfer coefficient
def PHI(z,  phi_o=0.08, alpha=0.2):
    return phi_o*np.exp(-alpha*z)

#PDE describing U(z,t)
def PDE_A(Uo,  i,  j,  m=0.1,  l=0.1):
    z = Z(i,  m=m)
    t = time(j,  l=l)
    return Uo/(1+(z-2*t)**2)

#Analytical solution to U(z,t)
def U(t, z=1):
    return 1/(1+(z-2*t)**2)
    
#Converts element index back to length
def Z(i,  m=0.1,  zo=0):
    return zo+i*m
    
#Converts element index back to time
def time(j,  l=0.1,  to=0):
    return to+j*l
    
#Creates an array of t values for 3D plot meshgrid
def tvalues(Tmx,  l=0.1):
    tvals = np.array([[i*l] for i in range (0,  len(Tmx))])
    return tvals
    
#Creates an array of z values for 3D plot meshgrid
def zvalues(Tmx,  m=0.1):
    zvals = np.array([[i*m] for i in range(0, len(Tmx[0]))])
    return zvals

#PDE describing temperature
def PDE_B(Tij, Tprev_ij,  Ts, v, l, m, phi):
    return (1-v*l/m-phi*l)*Tij + v*l*Tprev_ij/m + phi*l*Ts
    
#Plots analytical vs numerical solutions if using PDE U(z,t)
def plot_U(Umx, z=1, l=0.1):
    plt.figure(num=None, figsize=(10,  6), dpi=120, facecolor='w', edgecolor='k')
    trange = len(Umx)
    tvals = np.arange(0, trange*l,  l)
    #print('trange = %s \n Umx =  %s' % (trange,  Umx))
    Unum = np.array([Umx[j, z] for j in range(0, trange)])
    plt.plot(tvals, U(tvals),  label='Analytical')
    print('Uvals:',  Unum)
    plt.plot(tvals,  Unum,  label='Numerical')
    plt.title('U (z,t)')
    plt.ylabel('U (z=%s, t)' % z)
    plt.xlabel('Time')
    plt.legend()
    plt.grid()
    plt.show()
    #Plot square residuals
    plt.figure(num=None, figsize=(10,  6), dpi=120, facecolor='w', edgecolor='k')
    plt.title('Square Residuals vs Time')
    plt.plot(tvals,  (Unum - U(tvals))**2)
    plt.ylabel('(U_Numerical - U_Analytical)^2')
    plt.xlabel('Time')
    plt.grid()
    plt.show()
    SSR = np.sum((Unum - U(tvals))**2)
    print('SSR = %s' % SSR)

#Plots the temperature profile 
def plot_Tmx(Tmx, l=0.1, m=0.1, numLines=5):
    plt.figure(num=None, figsize=(10,  6), dpi=120, facecolor='w', edgecolor='k')
    zrange = len(Tmx[0])
    zvals = np.arange(0, zrange*m, m)
    
    #Plots the desired number of lines spaced equally through the dataset
    for i in range(0, len(Tmx),  int(len(Tmx)/numLines)):
        print('Tmx[%s] = %s' % (i,  Tmx[i]))
        plt.plot(zvals, Tmx[i],  label=('T(t=%ss)' % int(i*l)))
    
    plt.title('Temperature Profile (z,t)')
    plt.legend()
    plt.ylabel('Temperature (K)')
    plt.xlabel('Length (m)')
    plt.grid()
    plt.show()
    
#Plots the steady-state exit temperature for each alpha value
def plot_alpha_SS(totalTmx, alpha_vals, l=0.1,  m=0.1):
    plt.figure(num=None, figsize=(10,  6), dpi=120, facecolor='w', edgecolor='k')
    ss = len(totalTmx[0])-1
    exit = len(totalTmx[0][ss])-1
    alpha_Tmx = []
    for i in range(0, len(totalTmx)):
        print('totalTmx[%s] = %s ' % (i,  totalTmx[i][ss,  exit]))
        alpha_Tmx.append(totalTmx[i][ss,  exit])
    plt.plot(alpha_vals, alpha_Tmx,  color='r',  marker='o',  markersize=10)
    plt.title('Steady State Exit Temperature versus Alpha')
    plt.ylabel('Temperature (K)')
    plt.xlabel('Alpha')
    plt.grid()
    plt.show()

#Plots the steady-state exit temperature for each velocity value
def plot_v_SS(v_vals, ssTemps, l=0.1,  m=0.1):
    plt.figure(num=None, figsize=(10,  6), dpi=120, facecolor='w', edgecolor='k')
    plt.plot(v_vals, ssTemps,  color='r')
    plt.title('Steady State Exit Temperature versus Velocity')
    plt.ylabel('Temperature (K)')
    plt.xlabel('Velocity (m/s)')
    plt.grid()
    plt.show()

#3D Plot of T(z,t)
def HeatMap(Tmx,  l=0.1,  m=0.1):
    fig = plt.figure(figsize=(10,  6), dpi=120)
    ax = fig.add_subplot(111, projection='3d')
    zvals = zvalues(Tmx)
    tvals = tvalues(Tmx)
    trange = len(tvals)
    zrange = len(zvals)
    Z, T = np.meshgrid(zvals,  tvals)
    surf = ax.plot_surface(Z, T,  Tmx, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_title('Temperature as a Function of Length and Time')
    ax.set_xlabel('z (m)')
    ax.set_ylabel('t (s)')
    ax.set_zlabel('Temperature (K)')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

zmax = 3 #Maximum number of z-points desired
tmax = 60 #Current time
m = 0.1 #m
l = 0.1 #s
v = 0.1 #m/s
v_step = 0.01
v_max = 0.9
v_vals = []
To = 273.15+30 #K
Ts = 273.15+200 #K
alpha = 0.16 #First alpha
alpha_max = 0.3 #Last alpha
alpha_step = 0.01 #Increment of alpha increase
alpha_vals = []
k = 0
ssTemps = []
eqn = 'B'
run_alpha = True
#Determines the number of elements based on resolution and maximum values
zrange = int(zmax/m+1)
trange = int(tmax/l+1)

if (run_alpha == True):
    #3D Array containing each temperature profile solution for all alpha values
    totalTmx = np.array([[[0]*zrange]*trange]*int((alpha_max - alpha)/alpha_step + 2),  dtype=float)
else:
    #3D Array containing each temperature profile solution for all velocity values
    totalTmx = np.array([[[0]*zrange]*trange]*int((v_max - v)/v_step + 2),  dtype=float)

#Loop iterates through each alpha value
if (run_alpha == True):
    while (alpha <= alpha_max+0.0005):
        print('zrange = %s \n trange = %s' % (zrange,  trange))
        alpha_vals.append(alpha)
        
        #T[j][i] where T[j] is all z for a single t and T[j][i] is a specific temperature at t and z
        Tmx = np.array([[0]*zrange]*trange,  dtype=float)
     
        #Assert boundary condition where z=0 at any time
        for y in range(0,  trange):
            if (eqn == 'A'):
                Tmx[y,  0] = PDE_A(1, 0,  y,  l=l)
            if (eqn == 'B'):
                Tmx[y,  0] = To #BC
        #Assert initial condition where t=0 at any z
        for x in range(0,  zrange):
            if (eqn == 'A'):
                Tmx[0,  x] = PDE_A(1, x,  0,  m=m)
            if (eqn == 'B'):
                Tmx[0,  x] = Ts #IC


        #Create array of phi values (reused for each iteration)
        phi_array = np.array([0]*zrange,  dtype=float)
        for i in range(0,  zrange):
            phi_array[i] = PHI(i*m,  alpha=alpha)
            
        #Iterate through time
        for j in range(0,  trange-1):
            #Calculate new temperatures down length of HX
            for i in range(1,  zrange):
                phi = phi_array[i] #Phi(z)
                Tc = Tmx[j,  i] #T i,j
                Tbk = Tmx[j,  i-1] #T i-1,j
                #Integrates U(z,t)
                if (eqn == 'A'):
                    Tmx[j+1][i] = PDE_A(1,  i,  j,  m=m,  l=l)
                #Integrates T(z,t)
                if (eqn == 'B'):
                    Tmx[j+1][i] = PDE_B(Tc, Tbk,  Ts, v, l, m, phi)
        
        print('Appending totalTmx[%s]. Alpha = %s.' % (k,  alpha))
        totalTmx[k] = Tmx # Save each Tmx result
        #If PDE is temperature only
        if (eqn == 'B'):
            ssTemps.append((alpha,  Tmx[trange-1][zrange-1]))
            alpha += alpha_step
            k += 1
        #If PDE is U(z,t)
        else:
            plot_U(Tmx,  z=10,  l=l)
            alpha = 2*alpha_max

    #If the equation involved temperatures, plots corresponding information
    if eqn=='B':
        print('Steady state temperatures: \n',  ssTemps)
        plot_Tmx(totalTmx[0],  l=l,  m=m,  numLines=10)
        plot_alpha_SS(totalTmx, alpha_vals, l=l,  m=m)
        HeatMap(totalTmx[0],  l=l, m=m)
        
if (run_alpha == False):
    while (v <= v_max+0.0005):
        print('zrange = %s \n trange = %s' % (zrange,  trange))
        v_vals.append(v)
        
        #T[j][i] where T[j] is all z for a single t and T[j][i] is a specific temperature at t and z
        Tmx = np.array([[0]*zrange]*trange,  dtype=float)
     
        #Assert boundary condition where z=0 at any time
        for y in range(0,  trange):
            if (eqn == 'A'):
                Tmx[y,  0] = PDE_A(1, 0,  y,  l=l)
            if (eqn == 'B'):
                Tmx[y,  0] = To #BC
        #Assert initial condition where t=0 at any z
        for x in range(0,  zrange):
            if (eqn == 'A'):
                Tmx[0,  x] = PDE_A(1, x,  0,  m=m)
            if (eqn == 'B'):
                Tmx[0,  x] = Ts #IC


        #Create array of phi values (reused for each iteration)
        phi_array = np.array([0]*zrange,  dtype=float)
        for i in range(0,  zrange):
            phi_array[i] = PHI(i*m,  alpha=alpha)
            
        #Iterate through time
        for j in range(0,  trange-1):
            #Calculate new temperatures down length of HX
            for i in range(1,  zrange):
                phi = phi_array[i] #Phi(z)
                Tc = Tmx[j,  i] #T i,j
                Tbk = Tmx[j,  i-1] #T i-1,j
                #Integrates U(z,t)
                if (eqn == 'A'):
                    Tmx[j+1][i] = PDE_A(1,  i,  j,  m=m,  l=l)
                #Integrates T(z,t)
                if (eqn == 'B'):
                    Tmx[j+1][i] = PDE_B(Tc, Tbk,  Ts, v, l, m, phi)
        
        print('Appending totalTmx[%s]. Velocity = %s.' % (k,  v))
        totalTmx[k] = Tmx # Save each Tmx result
        #If PDE is temperature only
        if (eqn == 'B'):
            ssTemps.append(Tmx[trange-1][zrange-1])
            v += v_step
            k += 1
            
    print('SS Temps:',  ssTemps)
    print(len(v_vals))
    print(len(totalTmx))
    plot_v_SS(v_vals, ssTemps,  l=l,  m=m)
