# -*- coding: utf-8 -*-
"""
Created on Sat Mar  1 22:20:42 2014

@author: xiao
Date: 1 mars
email: xiao.wang@polytechnique.edu
"""

from numpy import * 
from numpy.linalg import * 

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
def f(d, u):
    D_u = getPositiveItemSet(u)
    res = 0
    for i in D_u:
        res += dot(V[i].T, V[d])
    return res/len(D_u)
    
#AUC loss function, sometimes called margin ranking loss
def AUC_loss_u(u):
    D_u = getPositiveItemSet(u)
    D = arange(m)
    D_u_bar = setdiff1d(D, D_u)
    res = 0
    for d in D_u:        
        for d_bar in D_u_bar:
            res += maximum(0, 1-f(d, u)+f(d_bar, u))
    return res
            
#SGD_AUC loss function
#selects a user, a positive item and a non-positive item at random, and makes a gradient step
def SGD_AUC_loss(u):
    #To be completed.