from math import exp
from numpy import * 
from numpy.linalg import * 
import time

filename = "/home/xiao/ProjetLibre/ml-5/u.data"
lamda = 2
alpha = 0.01
k = 5
#in fact m=5, n = 21 but we like to ignore indice 0
m = 6
n = 21
precision = 0.01
#Q is initialized as a set of random number
#Q = random.random((m))
Q = zeros(m)
#X[m][n]
ObservationX = zeros((m,n))

#U_old[m][k]
U = ones((m,k))
U_new = zeros((m,k))   
    
#V[n][k]
V = ones((n,k))
V_new = zeros((n,k))

#Z[m][n]
Z = zeros((m,n))

omegaSet = {}

def getAllOmega():
    for i in xrange(1, m):
        omegaSet[i] = getOmega(i)
        
#omega is the set of nonzero indices in X
#omega(i) is the set of nonzero indices in the ith row 
#the result is [omage, complement set of omage]
def getOmega(i):

#    T = ObservationX[i]
#    R1 = []
#    R2 = []
#    for j in xrange(1,n):
#        if T[j] != 0:
#            R1.append(j)
#        else:
#            R2.append(j)
#    return [R1, R2]

    T = ObservationX[i][1:]
    R1 = nonzero(T > 0)[0] + 1
    R2 = nonzero(T == 0)[0] + 1
    return [R1, R2]
  
#Xi = Ui*(Vp-Vq)t
def getX(i, p, q):
    return sum(inner(U[i][1:], (V[p][1:] - V[q][1:])))
    
#Yi = Ui*Vp^t -qi 
def getY(i, p):
    return sum(inner(U[i][1:], (V[p][1:]))) - Q[i]
#    return U[i][1:] * (V[p][1:].T) - Q[i] 
    
        
def derivativeU(i, j):
    derivative = lamda*U[i][j]
    temp = 0
    [omega, omega_comp] = omegaSet[i]
    for p in omega:
        for q in omega_comp:
            X = getX(i, p, q)
            Y = getY(i, p)
            A = 1/((1+exp(-X) + exp(-Y) + exp(-X-Y))**2)
            temp += A*(exp(-X)*(V[p][j] - V[q][j]) + exp(-Y)*V[p][j] + (2*V[p][j]-V[q][j])*exp(-X-Y))

    derivative -= temp
    return derivative

def derivativeV(k, j):
    derivative = lamda * V[k][j]
    temp = 0
    for i in xrange(1, m):
        [omega, omega_comp] = omegaSet[i]
        for p in omega:
            for q in omega_comp:
                X = getX(i, p, q)
                Y = getY(i, p)
                A =1/((1+ exp(-X) + exp(-Y) + exp(-X-Y))**2)
                if k in omega:
                    temp += (U[i][j]*(exp(-X)+ exp(-Y) + 2*exp(-X-Y)))*A
                elif k in omega_comp:
                    temp -= (U[i][j]*(exp(-X)+exp(-X-Y)))*A
    derivative -= temp
    return derivative
     
def getNewU():
    for i in xrange(1, m):
        for j in xrange(1, k):
            U_new[i][j] = U[i][j] - alpha*derivativeU(i,j)


def getNewU_stochastic():
    i = random.randint(1,m-1)
    for j in xrange(1,k):
        U_new[i][j] = U[i][j] - alpha*derivativeU(i,j)


def getNewV_stochastic():
    i = random.randint(1,n-1)
    for j in xrange(1,k):
        V_new[i][j] = V[i][j] - alpha*derivativeV(i,j)


def getNewV():
    for i in xrange(1, n):
        for j in xrange(1, k):
            V_new[i][j] = V[i][j] - alpha*derivativeV(i,j)

#U = U_new
def copieU():
    U = U_new.copy()

#V = V_new
def copieV():
    V = V_new.copy()
    
def printU():
    print U[:, 1:]

def printV():
    print V[:, 1:]

def printX():
    print ObservationX[:, 1:]

#compute normeF square of a matrix U of dimension m*k
def normeFS(S):
    return (S**2).sum()
    
def printLocalAUC():
    Z = dot(U,V.T)
    accu_correct = 0
    accu_total = 0
    res = 0.0
    for i in xrange(1, m):
        [omega, omega_comp] = omegaSet[i]
        accu_total += array(omega).size * array(omega_comp).size

        for p in omega:
            for q in omega_comp:
                if ( Z[i][p] > Z[i][q] and Z[i][p] > Q[i]):
                    accu_correct += 1

    res = (accu_correct+0.0) / accu_total
    print ("localAUC: ",res)
    return res


start = time.clock()
#read ratings from the file
f = open(filename, 'r')

for line in f:
    nums = [int(x) for x in line.split()]
    client = nums[0]
    film = nums[1]
    rating = nums[2]
    ObservationX[client][film] = rating
getAllOmega()
getNewU()
getNewV()
count = 0
distanceU = norm(U_new- U)
distanceV = norm(V_new- V)
#newDistanceU and newDistanceV are initialized are big numbers
#in order to enter the first while loop
newDistanceU = precision*10 + distanceU
newDistanceV = precision*10 + distanceV

while( abs(newDistanceU) > precision 
or abs(newDistanceV) > precision):
    count += 1
    print ("count :%d"% count)

    (distanceU,distanceV) = (newDistanceU,newDistanceV)


    U = U_new.copy()
    V = V_new.copy()

    getNewU_stochastic()
    getNewV_stochastic()

    newDistanceU = norm(U_new-U)
    newDistanceV = norm(V_new-V)

    print newDistanceU
    print newDistanceV

print "\n\n"
print abs(newDistanceU)
print abs(newDistanceV)
#printX()

printLocalAUC()
print ("erased time :", time.clock() - start)
