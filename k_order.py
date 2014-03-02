# -*- coding: utf-8 -*-
"""
Created on Sat Mar  1 22:20:42 2014

@author: xiao
Date: 1 mars
email: xiao.wang@polytechnique.edu
"""

from numpy import * 
from numpy.linalg import * 
from math import sqrt
from random import choice

#######################################
###  Initialisation des parametres  ###
#######################################

#m : number of users
m
#M : set of users
#D : set of items
n
#X : matrix of training data, of dimension |M|*|D|
X = zeros((m,n))

#V : learning matrix, of dimension |M|*|D|
V = zeros((m,n))

#Omega is a map that stores the set of positive ranked items for each user 
Omega={}

#lambda
learningSpeed = 1.0
#constraint of weight
C=3

#finds and stores positive ranked items for all user
#store in the dictionary Omega
#The function and the data structure Omega are created to avoid repeated calculation
#for each user u, Omega[i] is the set of indices of positive ranked items
def generateOmega():
    for i in range(m):
        Omega[i]= nonzero(X[i]>0)[0]  # find the indices of the nonzero elements
        
    
#returns set of positive ranked items for user u
def getPositiveItemSet(u):
    return nonzero(X[i]>0)[0]    

#Learning set: V, of dimension m * len(D)
#scoresOfU(int u): function that returns the vector of all item scores for the user u
#lossFunction

#factorized model to learn the ranking 
def f_d(d, u):
    D_u = getPositiveItemSet(u)
    res = 0
    for i in D_u:
        res += dot(V[i].T, V[d])
    return res/len(D_u)
    
def f_bar_d(bar_d, u):
    D_u = getPositiveItemSet(u)
    D = arange(m)
    D_u_bar = setdiff1d(D, D_u)

    res = 0.0
    for j in D_u_bar:
        res += dot(V[j].T,V[bar_d])
    return res/(len(D_u_bar))   
    

def g(u, d, bar_d):
''' g(u,d,bar_d) = max(0, 1-f_d(u)+f_bar_d(u))'''
    return maximum(0, 1-f_d(d,u)+f_bar_d(bar_d,u))
    
#AUC loss function, sometimes called margin ranking loss
def AUC_loss_u(u):
    D_u = getPositiveItemSet(u)
    D = arange(m)
    D_u_bar = setdiff1d(D, D_u)
    res = 0.0
    for d in D_u:        
        for bar_d in D_u_bar:
            res += g(u,d,bar_d)
    return res

def AUC_loss():
'''AUC loss function equals the sum of AUCLoss for each u in U.'''
    res = 0.0
    for u in range(m):
        res += AUC_loss_u(u)
    return res
                
        
#SGD_AUC loss function
#selects 1) a user, 2) a positive item and 3) a non-positive item at random, 
#and makes a gradient step
def SGD_AUC_loss(u):
    #To be completed.

def pick_positive_item(mean, std_deviation):
    '''K-os algorithm for picking a positive item'''
    ''' ???????????????????????????????????????? '''
    return k
    
def K_os_AUC_loss():
    '''Algorithm 3 in paper'''
    iterationTime = 2000
    while(iterationTime -- ):
        #1. pick a random user u
        #use random.choice()
        u = choice(U)
         
        #pick a positive item d
        d = pick_positive_item(0, 1/sqrt(m))
        
        #pick a random item bar_d
        #use random.choice to select randomly an element from a set
        D_u = getPositiveItemSet(u)
        D = arange(m)
        D_u_bar = setdiff1d(D, D_u)
        
        bar_d = choice(D_u_bar)
                
        if f_d_bar(u)>f_d(u)-1:
            V[u,d]+= learningSpeed*sum(V[u][D_u])*V[u,d]/len(D_u)
            V[u,bar_d]-=learningSpeed*sum(V[j][D_u_bar]*V[u,bar_d])/len(D_u_bar)
            if linalg.norm(V[:,d]) > C:
                dnum = linalg.norm(V[:,d])
                V[:,d]=C*V[:,d]/dnum


                
            
            
        
    
        
    
