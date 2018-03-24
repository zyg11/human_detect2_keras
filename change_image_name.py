import os
path_name='E:/keras_data/face/face-other/face-other'
i=0
for item in os.listdir(path_name):
    os.rename(os.path.join(path_name,item),os.path.join(path_name,('3.'+str(i)+'.jpg')))#0或者其他的
    i+=1