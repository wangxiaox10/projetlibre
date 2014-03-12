# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 11:25:58 2014

@author: xiao

xiao.wang@polytechnique.edu


There are two systems of indexes for each user and for each item 
Because in the original file, uesr/item id is not consecutive
This may bring problem of empty colomne/row in matrix if we construct 
a matrix of dimension(max_user_id, max_item_id)

That's why the first thing we shoud do is to transfer user/item id to matrix index
and functions for reverse translation. 


This file reads from original file and stores data in matrix format
There are five files:

1) file that indicates corresponding user_index_in_matrix and user_id_in_file
format: 
user_index_in_matrix user_index_in_file

2) file that indicates corresponding user_id_in_file and user_index_in_matrix
format:
user_index_in_file user_index_in_matrix

3) file that indicates corresponding item_index_in_matrix and item_id_in_file
format:
item_index_in_matrix item_id_in_file

4) file that indicates corresponding item_id_in_file and item_index_in_matrix
item_id_in_file item_index_in_matrix

5) file that indicates each rating
format:
user_index_in_matrix item_index_in_matrix
"""

"""complete path of the file to read"""

inputfile = "/home/xiao/ProjetLibre/ml-5/u.data"
#saves every positive item in matrix
#every line in format:
#user_index_in_matrix item_index_in_matrix rate
matrixInfo = "matrix/matrixInfo"

#user_identifiant user_index_in_matrix
userIdIndex = "matrix/userIdIndex"

#user_identifiant user_index_in_matrix
userIndexId = "matrix/userIndexId"

itemIndexId = "matrix/itemIndexId"
#item_identifiant item_index_in_matrix
itemIdIndex = "matrix/itemIdIndex"

f_input = open(inputfile, 'r')

f_matrixInfo = open(matrixInfo, 'w')
f_userIdIndex = open(userIdIndex, 'w')
f_userIndexId = open(userIndexId, 'w')
f_itemIdIndex = open(itemIdIndex, 'w')
f_itemIndexId = open(itemIndexId, 'w')

#user_index: index for next visited_user
#item_index: index for next visited_item

#Initialisation
user_index=0;
item_index=0;

#set of visited users
users={}
#user_id : usre_index
users_id_index={}
users_index_id={}

#set of visited items
items={}
#item_id: item_index
items_id_index={}
items_index_id={}

for line in f_input:
    nums = [int(x) for x in line.split()]
    (user_id, item_id) = (nums[0:2])
    
    if( user_id not in users_id_index):
        '''update map users_id_index, users_index_id, user_index'''
        users_id_index[user_id] = user_index
        users_index_id[user_index] = user_id
        user_index += 1
        
    if( item_id not in items_id_index):
        '''update items_id_index, items_index_id, item_index'''
        items_id_index[item_id] = item_index
        items_index_id[item_index] = item_id
        """index for next item"""
        item_index += 1
    #print author_index-1, item_index-1, 1
    f_matrixInfo.write("%d %d %d\n" %(users_id_index[user_id] ,items_id_index[item_id], 1))
    
for user_id in users_id_index:
    f_userIdIndex.write("%d %d\n" %(user_id, users_id_index[user_id]))

for user_index in users_index_id:
    f_userIndexId.write("%d %d\n" %(user_index, users_index_id[user_index]))
        
for item_id in items_id_index:
    f_itemIdIndex.write("%d %d\n" %(item_id, items_id_index[item_id]))
    
for item_index in items_index_id:
    f_itemIndexId.write("%d %d\n" %(item_index, items_index_id[item_index]))
    
 

print ("# users m :%d, # items n=%d" %(len(users_id_index), len(items_id_index)))
