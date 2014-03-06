# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 20:55:48 2014

@author: xiao
"""
from numpy import * 
from random import *
from numpy.linalg import * 
import matplotlib.pyplot as plt

m=5
n=20
X=zeros((m,n))
#read ratings from the file
filename = "/home/xiao/ProjetLibre/ml-5/u.data"
f = open(filename, 'r')
for line in f:
    nums = [int(x) for x in line.split()]
    client = nums[0]
    film = nums[1]
    rating = nums[2]
    X[client-1][film-1] = 1


lamda = 0.5
precision = 0.001
C=3
K=3
k=1
V=zeros((m,n))
for i in range(m):
    for j in range(n):
        V[i,j]=random()

Omega={}
#print X
#print V

#########################################
####   get posotive items of user u  ####
#########################################

def getDu():
    '''get posotive items of user u'''
    '''return an ensemble of set of positive items for each user'''
    for i in range(m):
        t = X[i,:]
        Du=nonzero(t>0)[0]
        Omega[i]=Du
        
'''factorized models'''
def f_d(d,u):
    Du=Omega[u]
    t=V[:,Du]
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
    return maximum(0, 1-f_d(d,u)+f_bar_d(bar_d,u))
    
    
###################################################
####  making one gradient step to matrix V     ####
####  To minimize g=max(0, 1-fd(u)+f\bar d(u)) ####
###################################################
    
def derivativeD(u,d,bar_d):
    D_u = Omega[u]
#    print V[u,D_u]
    s = V[u,D_u].sum(axis=0)
    return (-V[u,d]+V[u,bar_d]-s)/len(D_u)

def derivativeBarD(u,d,bar_d):
    D_u = Omega[u]
    s = V[u,D_u].sum(axis=0)
    return s/len(D_u)
    
########################################################
####    localAUC Gradient Descent Algorithm         ####
########################################################
####  We hardly use it, because it's too costly.    ####
########################################################
    
def localAUC_u(u):
    '''we hardly use it, because it's too costly. '''
    D_u=Omega[u]
    D = arange(n)
    D_u_bar = setdiff1d(D, D_u)
    
    res=0
    for d in D_u:
        for bar_d in D_u_bar:
            res+=g(u,d,bar_d)
    return res

def localAUC():
    res=0
    for u in range(m):
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
    
    d = choice(D_u)
    bar_d = choice(D_u_bar)
    return (u,d,bar_d)
            
    
def gda_localAUC_oneStep():
    (u,d,bar_d)=chooseRandoms()
    if(f_d(d,u)<f_bar_d(bar_d,u)+1):
        V[u,d] -= lamda * derivativeD(u,d,bar_d)
        V[u,bar_d] -= lamda * derivativeBarD(u,d,bar_d)

def gda_localAUC():
    i=0
    
    print "iteration:",i
    gda_localAUC_oneStep()
    
    distanceV[i]=localAUC()
    i+=1        
    
    while i<iterationTime:
        print "iteration:",i
        gda_localAUC_oneStep()
        distanceV[i]=localAUC()
        i+=1
    
#    
iterationlimits = 20
distanceV=empty(iterationlimits)
getDu()

testRank()
