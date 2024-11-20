import torch 
from torch import nn
import numpy as np
import math
#self attention divided into 3 Query,key,values so make the 3 matrices
L=4
q,k=8,8
v=8
#as we are not using any dataset generate the query,key matrices with the random.randn function 
query_matrix=np.random.randn(L,q)
key_matrix=np.random.randn(L,k)
value_matrix=np.random.randn(L,v)
 
#as for the attention head ,the first step is mutliplication of query and key matrix and one of the matrix should be transpose 
Attention=np.matmul(query_matrix,key_matrix.T)/math.sqrt(k)
#write a function for the softmax
def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=-1).T
#make a masked matrix which is used for 
masked_matrix=np.tril(np.ones((L,L)))
# print(masked_matrix)
masked_matrix[masked_matrix==0]=-np.infty
masked_matrix[masked_matrix==1]=0
# print(masked_matrix)
Attention_score=Attention+masked_matrix
# print(Attention_score)
scaled=softmax(Attention_score)
# print(scaled)
new_score=np.matmul(scaled,value_matrix)
print(new_score)