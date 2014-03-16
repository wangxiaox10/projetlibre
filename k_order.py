# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 20:55:48 2014

@author: xiao
"""
from numpy import * 
from math import * 
from numpy.linalg import * 
from multiprocessing import pool
import matplotlib.pyplot as plt

###################################
#### Original data             ####
###################################
#m number of users
m=100
#number of items 
n=1238
#matrix of observation
X=zeros((m,n))
#m is the dimension of the embedding 
#usually like 50 or 100
dim_embedding = 10


###################################
#### Learning parameters       ####
###################################
lamda = 0.05
precision = 0.001
C=1
K=2
k=1
iterationlimits=500000

distanceV=empty(100)
        
#########################################
#### Intermediaire data structures   ####
#########################################
Omega={}
#the set of set of non-positive items for each user
Omega_comp={}



#V is matrix of dimension dim_embedding * n 
V = random.normal(0, 1.0/sqrt(dim_embedding), dim_embedding * n)
V = reshape(V, (dim_embedding, n))

#for i in range(dim_embedding):
#    for j in range(n):
#        V[i,j]=random()
#    if( linalg.norm(V[:,i])>C):
#        V[:,i]=C*V[:,i]/linalg.norm(V[:,i])
print "finish initialize V"
        

#########################################
####    read ratings from the file   ####
#########################################
def readDataFromFile(fadress):
    '''the rating is binaire'''
    filename = fadress
    f = open(filename, 'r')
    for line in f:
        nums = [int(x) for x in line.split()]
        if len(nums)<3:
            print nums
        client = nums[0]
        film = nums[1]
#        rating = nums[2]
        X[client-1][film-1] = 1

#########################################
####   get posotive items of user u  ####
#########################################

def getDu():
    '''get posotive items of user u'''
    '''return an ensemble of set of positive items for each user'''
    for i in range(m):
        t = X[i,:]
        D_u=nonzero(t>0)[0]
        Omega[i]=D_u
    
        D = arange(n)
        D_u_bar = setdiff1d(D, D_u)
        Omega_comp[i] = D_u_bar
        
        
'''factorized models'''
def f_d(d,u):
    Du=Omega[u]
    global V
    t=V[:,Du]
#    print t
    vec_items_pos_ranked = t.sum(axis=1)
    return dot(vec_items_pos_ranked.T, V[:,d])/len(Du)

def f_bar_d(bar_d, u):
    return f_d(bar_d,u)
    
def f(u):
    res = empty(n)
    for d in range(n):
#        print "d:",d
        res[d]=f_d(d,u)
    return res
        
def g(u, d, bar_d):
    ''' g(u,d,bar_d) = max(0, 1-f_d(u)+f_bar_d(u))'''
#    print "d",d,"u",u,"bar_d",bar_d,"f_d",f_d(d,u),"f_bar_d",f_bar_d(bar_d,u)
#    print maximum(0, 1-f_d(d,u)+f_bar_d(bar_d,u))
    return maximum(0, 1-f_d(d,u)+f_bar_d(bar_d,u))
    
    
###################################################
####  making one gradient step to matrix V     ####
####  To minimize g=max(0, 1-fd(u)+f\bar d(u)) ####
###################################################
    
#one gradient step to each element in matrix V
def derivativeOneStep(u,d, bar_d):
    '''for element Vpq, p in range(m), q in range( #items)'''
    '''for each p:'''
    '''if q \in Du and q != d: step = -Vpd + Vpbar_d'''
    '''if q \in Du and q = d:  step = -Vpq + Vpbar+d + \sum_{i \in Du} Vpi'''
    '''if q not \in Du and q != bar_d: step = 0'''
    '''if q not \in Du and q == bar_d: step = \sum_{i \in Du} Vpi'''
    print "start computing derivatives", u, d, bar_d
    
    D_u = Omega[u]
    step = zeros((dim_embedding, n))
    temp = 0
    for q in D_u:
        step[:, q] = -V[:,d] + V[:, bar_d]
        temp += V[:,q]
    step[:,d] += temp
    step[:,bar_d] += temp

    print "finishing computing step"
    
    return step
    


def derivativeD(u,d,bar_d):
    D_u = Omega[u]
#    print V[u,D_u]
    '''problem here! V[u, D_u]'''
    s = V[u,D_u].sum(axis=0)
    return (-V[u,d]+V[u,bar_d]-s)/len(D_u)

def derivativeBarD(u,d,bar_d):
    D_u = Omega[u]
    '''problem here! V[u, D_u]'''
    s = V[u,D_u].sum(axis=0)
    return s/len(D_u)
    
########################################################
####    localAUC                                    ####
########################################################
####  We hardly use it, because it's too costly.    ####
########################################################
    
def localAUC_u(u):
    '''we hardly use it, because it's too costly. '''
    D_u=Omega[u]
#    D = arange(n)
#    D_u_bar = setdiff1d(D, D_u)
    D_u_bar = Omega_comp[u]
    
    res=0
    
    fvalue = f(u)
    for d in D_u:
        f_d_u = fvalue[d]
        for bar_d in D_u_bar:
            f_bar_d_u = fvalue[bar_d]
            res += maximum(0, 1-f_d_u +f_bar_d_u )
#            res+=g(u,d,bar_d)
    return res

def localAUC():
    res=0
    for u in range(m):
        print "localAUC u:", u, "res:", res
        res+=localAUC_u(u)
    return res


################################################
####    Implementation of algorithm 1       ####
####   k-os for picking a positive item     ####
################################################

def pickPositiveItem():
    '''Algorithm 1 in paper'''
    
    '''pick a user at random from the training set'''
    u = randint(0,m-1)
    '''pick i=1,...,K positive items d_i\in D_u'''
    seq = sample(Omega[u],K)
    '''compute f_di(u) for each i'''
    f_seq = f_d(seq,u)
    '''sort the scores by descending order'''
    f_seq.sort()
    f_seq=f_seq[::-1]
    '''pick a position k \in 1,...K using the distribution'''
    max_k_order = f_seq[k-1]
    
    for i in range(K):
        if f_d(seq[i],u) == max_k_order:
            return (u,seq[i])
            
################################################
####    Implementation of algorithm 2       ####
####           k-os WARP loss               ####
################################################
            
def k_os_WARP_one_step():
    '''pick a positive item d using pickPositiveItem()'''
    (u,d)=pickPositiveItem()
    N=0
    
    '''pick a random item bar_d'''
    D_u = Omega[u]
    D = arange(n)
    D_u_bar = setdiff1d(D, D_u)
    
    bar_d = choice(D_u_bar)
    N+=1
    
    '''start gradient descent learning'''
    while f_bar_d(bar_d,u) <= f_d(d,u)-1 and N < n-len(D_u):
        bar_d = choice(D_u_bar)
        N+=1
    if f_bar_d(bar_d,u)>f_d(d,u)-1 :
        coe = bigPhi_top((n-len(D_u))/N)
        '''update V_ud and V_u\bar d'''
        V[u,d] -= lamda * coe * derivativeD(u,d,bar_d)
        V[u,bar_d] -= lamda * coe * derivativeBarD(u,d,bar_d)
        
        '''Project weights to enforce constraints '''
        '''if ||Vi||>C'''
        '''then set Vi <- (C*Vi)/||Vi||'''
    if( linalg.norm(V[:,d])>C):
        V[:,d]=C*V[:,d]/linalg.norm(V[:,d])
    if( linalg.norm(V[:,bar_d])>C):
        V[:,d]=C*V[:,bar_d]/linalg.norm(V[:,bar_d])
        
def k_os_WARP():
    iterationTime=0    
    while(iterationTime<iterationlimits):
        k_os_AUC_one_step()
        '''register the results in a vector for figuration'''
        distanceV[iterationTime]=localAUC()
        iterationTime+=1

################################################
####    Implementation of algorithm 3       ####
####          k-os AUC loss                 ####
################################################

def k_os_AUC_one_step():
      
    '''pick a positive item d using pickPositiveItem()'''
    (u,d)=pickPositiveItem()
    
    '''pick a random item bar_d'''
    D_u = Omega[u]
    D = arange(n)
    D_u_bar = setdiff1d(D, D_u)
    
    bar_d = choice(D_u_bar)
    
    if f_bar_d(bar_d,u) > f_d(d,u)-1:
#        print "oui!"
        '''one gradient step'''
        V[u,d] -= lamda * derivativeD(u,d,bar_d)
        V[u,bar_d] -= lamda * derivativeBarD(u,d,bar_d)
        
        '''Project weights to enforce constraints'''
        if( linalg.norm(V[:,d])>C):
            V[:,d]=C*V[:,d]/linalg.norm(V[:,d])
        if( linalg.norm(V[:,bar_d])>C):
            V[:,d]=C*V[:,bar_d]/linalg.norm(V[:,bar_d])
            
def k_os_AUC():
    '''initialize model parameters'''
    iterationTime=0    
    while(iterationTime<iterationlimits):
        k_os_AUC_one_step()
        distanceV[iterationTime]=localAUC()
        iterationTime+=1
     

#########################################################################
#### Phi function converts the rank of a positive item d to a weight ####
#########################################################################

#this bigPhi function makes WARP equivalent to AUC.
def bigPhi_AUC(eta):
    return C*eta

#This bigPhi focus on items on the top of list    
def bigPhi_top(eta):
    res = 0.
    for i in range(eta):
        res += 1/i
    return res
    
        
        
#####################################################################
####  the more user likes an item, the score ranked is higher    ####
#####################################################################
        
#equation (2)
def rank(d,u):
    D_u = Omega[u]
    D = arange(n)
    D_u_bar = setdiff1d(D, D_u)
    
    fu = f_d(d,u)

    count = 0    
    for bar_d in D_u_bar:
        if fu >= 1 + f_bar_d( bar_d, u):
            count += 1
    return count
    
def testRank():
    for u in range(m):
        print "u:", u
        D_u = Omega[u]
        for d in D_u:
            print "d:",d, rank(d,u)

####################################################################
####    General loss function function                          ####
####    Definition see the right-up corner in the 3rd page      ####
####################################################################

def k_os_lossFunction_u(u):
    '''rank(d,u), where d is item in the kthe position'''
    '''sort the scores by descending order'''
    seq = Omega[u]    
    f_seq = f_d(seq,u)    
    f_seq.sort()
    f_seq=f_seq[::-1]
    '''pick a position k \in 1,...K using the distribution'''
    max_k_order = f_seq[k-1]
    
    for i in range(K):
        if f_d(seq[i],u) == max_k_order:
            order = seq[i]
            return bigPhi(rank(order, u))
            
def k_os_lossFunction():
    res=0
    for u in range(m):
        res += k_os_lossFunction_u()
    return res
    
######################################################
####  Implementation  of regular AUC algorithm    ####
######################################################
    
def chooseRandoms():
    u = randint(0,m-1)
    D_u = Omega[u]
    while(len(D_u)==n):
        u = randint(0,m-1)
        D_u = Omega[u]
        
    D = arange(n)
    D_u_bar = setdiff1d(D, D_u)
    
    i = randint(0, len(D_u)-1)
    j = randint(0, len(D_u_bar)-1)
    d = D_u[i]
    bar_d = D_u_bar[j]
    return (u, d, bar_d)
#    d = random.choice(D_u)
#    bar_d = random.choice(D_u_bar)

            
    
def gda_localAUC_oneStep():
    (u,d,bar_d)=chooseRandoms()
    if(f_d(d,u)<f_bar_d(bar_d,u)+1):
        global V
        V -= lamda * derivativeOneStep(u, d, bar_d)
        for i in range (n):
            if( linalg.norm(V[:,i])>C):
               V[:,i]=C*V[:,i]/linalg.norm(V[:,i])
#        V[u,d] -= lamda * derivativeD(u,d,bar_d)
#        V[u,bar_d] -= lamda * derivativeBarD(u,d,bar_d)

def gda_localAUC():
    i=0
    
    print "iteration:",i
    gda_localAUC_oneStep()
    
    distanceV[i]=localAUC()
    print distanceV[i]
    i+=1        
    
    while i<iterationlimits:
        print "iteration:",i
        gda_localAUC_oneStep()
        if i % 5000 == 0:
            distanceV[i/5000] = localAUC()
#        distanceV[i]=localAUC()
        i+=1
#    plot(distanceV)
    
    
#    
    
######################################################
####               test normal AUC                ####
######################################################

#after iterations show localAUC
fadress = "/home/xiao/ProjetLibre/matrix/matrixInfo"
readDataFromFile(fadress)
getDu()
gda_localAUC()
