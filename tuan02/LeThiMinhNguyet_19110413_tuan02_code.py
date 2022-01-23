#%%bai2
array=[1, -15,3,21,0,5]
list=[]
for i in range(0,len(array)):
    if array[i]%5==0:
        list.append(array[i])
print(list)
#%%bai3
import numpy as np
arr1=np.array([-2, 6, 3, 10, 15, 48])
print(arr1[2:5:2])
print(arr1[1::2])
print(arr1[3:])
print(arr1[6:2:-1])
#%%bai4
arr=np.array([4,3,5,6])
a=[]
def sapxep(mang, SapXepTang=True):
    if SapXepTang==True:
        a=sorted(mang)
    else:
        a=sorted(mang, reverse=True)
    return a
print(sapxep(arr))
