import numpy as np
import matplotlib.pyplot as plt
import copy
import os
import re
#+================================================================================
'''读取txt文件,存为数组并且取出前两列的数据'''

def txt_read(file):
    data = open(file)
    txt = []

    for line in data:
        txt.append(line.strip())

    plot = []

    for i in range(len(txt)):
        img= np.array(txt[i].split()).astype(float)
        plot.append(img)

    plot = np.array(plot)

    return plot[:,0:2]#切片
#=====================================================================================
'''处理数据，拿出符合要求的波段'''

def deal_data(input):
    wave = []
    intensity = []

    try:
        input.shape[1] != 2
    except:
        print('The data format is wrong!')

    for i in range(input.shape[0]):
        wavelen = input[i,0]
        flux = input[i,1]
        if 6 <= wavelen <= 9:
            wave.append(wavelen)
            intensity.append(flux)

    new = np.zeros([len(wave),2])
    for i in range(len(wave)):
        new[i,0] = wave[i]
        new[i,1] = intensity[i]
    
    return new
#+==========================================================================
'''修改数据分辨率，然后重新储存'''

def change_res(input_):
    if 0.0004<=(input_[2,0] - input_[1,0])<= 0.0006:
        wave_ = []
        intensity_ = []
        for i in range(0,input_.shape[0],4):
            wave_.append(input_[i,0])
            intensity_.append(input_[i,1])
        new_ = np.zeros([len(wave_),2])
        for j in range(len(wave_)):
            new_[j,0] = wave_[j]
            new_[j,1] = intensity_[j]
        return new_
            
    if 0.0009<=(input_[2,0] - input_[1,0])<=0.0011:
        wave_ = []
        intensity_ = []
        for i in range(0,input_.shape[0],2):
            wave_.append(input_[i,0])
            intensity_.append(input_[i,1])
        new_ = np.zeros([len(wave_),2])
        for j in range(len(wave_)):
            new_[j,0] = wave_[j]
            new_[j,1] = intensity_[j]       
        return new_

    if 0.0019<=(input_[2,0] - input_[1,0])<= 0.0021:
        new_ = input_
        return new_
#=========================================================================


def get_filePath(path):
    '''
    input: 文件路径path
    '''
    file_or_dir = os.listdir(path)
    dir = []

    for file_dir in file_or_dir:
        file_or_dir_path = os.path.join(path,file_dir)
        dir.append(file_or_dir_path)
    return dir

#===============================================================================

path = 'D:\\data\\IR_data\\' #原始数据存放位置

file_path = get_filePath(path)

new_path = 'D:\\data\\new_path\\'

if not os.path.exists(new_path):
     os.makedirs(new_path)  

for file in file_path:
    img  = txt_read(file)
    result = deal_data(img)
    new_data = change_res(result)
    name = ''.join(re.findall(r'IR_data\\(.+?).txt', file))
    np.save(new_path+str(name)+'.npy',new_data)
