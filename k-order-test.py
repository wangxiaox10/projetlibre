# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 16:41:40 2014

@author: xiao
"""

from k_order import *
#number of items to recommand
p=2
fadress = "/home/xiao/ProjetLibre/matrix/matrixInfo"

readDataFromFile(fadress)
getDu()
recommendationListe = zeros((m,p))

############################################
####    We need to recommend top items  ####
############################################


#k=1
#recommend top p items for user u
def recommendItems_u(u, p):
    #initialize recommendation items to be -1: null
    res = zeros(p)-1
    D_bar_u = Omega_comp[u]
    r = f_bar_d(D_bar_u, u)
    
    indexOrder = argsort(r)
    indexOrder = indexOrder[::-1]

    if len(indexOrder) >= p:
        res = indexOrder[:p]
    else:
        res[:len(indexOrder)] = indexOrder
    return res

#recommend top p items for all m users
def recommendItems(p):    
    for u in range(m):
        r = recommendItems_u(u, p)
        recommendationListe[u,:] = r
    
def f_test(x):
    return x**2 - 3*x
    
def test():
    a = arange(5)
    b = f_test(a)
    c = argsort(b)
    c = c[::-1]
    return c
    
#show 
def showRecomms():
    for u in range(m):
        print "u:", u, ",",recommendationListe[u,:]
        
k_os_AUC()
recommendItems(p)
showRecomms()


######################################################
####    We need to recommend most relavent users  ####
######################################################


######################################################
####               test normal AUC                ####
######################################################

######################################################
####               test normal WARP               ####
######################################################

######################################################
####               test K-os AUC                  ####
######################################################

######################################################
####                test k-os WARP                ####
######################################################
