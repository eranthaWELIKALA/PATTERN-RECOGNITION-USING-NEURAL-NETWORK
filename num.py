import os
from scipy import misc
from numpy import *
# array = os.walk('test/')
# for root_dir,file_dir,files_list in array:
# 	print(root_dir,file_dir,len(files_list))	
array = os.listdir('test/')
newlist = []
for names in array:
    if names.endswith(".png"):
        newlist.append(names)
print (newlist)

fin = []
itr = 1
for i in newlist:
	fin.append([itr,len(misc.imread('test/'+i).flatten())])
	itr += 1
random.shuffle(fin)
print(fin)