import scipy as sp
from scipy.interpolate import interp1d
from scipy.signal import medfilt
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from scipy.signal import argrelmin
import copy

# read txt files, return numpy array

def txt_read(file):
    data = open(file)
    txt = []

    for line in data:
        txt.append(line.strip())

    plot = []

    for i in range(len(txt)):
        img = np.array(txt[i].split()).astype(float)
        plot.append(img)

    plot = np.array(plot)

    return plot
#=====================================================================================


def get_filePath(path):
    '''
    input: 文件路径path
    '''
    file_or_dir = os.listdir(path)
    dir = []

    for file_dir in file_or_dir:
        file_or_dir_path = os.path.join(path, file_dir)
        dir.append(file_or_dir_path)
    return dir

#===============================================================================


def interpolate_nan(x, y):
    x_not_nan = x[~np.isnan(y)]
    y_not_nan = y[~np.isnan(y)]
    f = sp.interpolate.interp1d(x_not_nan, y_not_nan)
    return f(x)


def fit_polynomial(x, y, degree):
    coeffs = np.polyfit(x, y, degree)
    f = np.poly1d(coeffs)
    return f



path = 'D:\\data\\linshi\\'  # 扣除连续谱之前的数据位置

file_path = get_filePath(path)

restore_path = 'D:\\data\\result_data\\'  # 扣除连续谱之后的数据位置
pic_path = 'D:\\data\\img_path\\after_cut_cont\\'
compare = 'D:\\data\\pic_path\\'

if not os.path.exists(restore_path):
    os.makedirs(restore_path)

if not os.path.exists(pic_path):
    os.makedirs(pic_path)


for file in file_path:
    name = ''.join(re.findall(r'linshi\\(.+?).txt', file))
    img = txt_read(file)
    wave = img[:, 0]
    # time = np.round(time, 2)
    flux = img[:, 1]
    flux_raw = copy.deepcopy(flux)

# emissions对应需要屏蔽的发射线位置，如果扣除连续谱后需要保留发射线，则需要将emissions和mask_intervals合并。
emissions = np.array([[6.98, 7.01], [7.65, 7.68]])  

# mask_intervals对应需要屏蔽的UIE波段位置
mask_intervals = np.array([[6.21, 6.38], [7.94, 8.24]])


# for interval in emissions:
#     start, end = interval
#     start_index = np.argmin(np.abs(wave - start))
#     end_index = np.argmin(np.abs(wave - end))
#     flux[start_index:end_index + 1] = np.nan

#     emission_signal = interpolate_nan(wave, flux)
#     fit_emission = fit_polynomial(wave, emission_signal, 200)
#     cont_emission = fit_emission(wave)
#     flux[start_index:end_index + 1] = cont_emission[start_index:end_index + 1]

flux_emission = copy.deepcopy(flux)

for interval in mask_intervals:
    start, end = interval
    start_index = np.argmin(np.abs(wave - start))
    end_index = np.argmin(np.abs(wave - end))
    flux[start_index:end_index + 1] = np.nan

interpolated_signal = interpolate_nan(wave, flux)
fit = fit_polynomial(wave, interpolated_signal, 10)
cont = fit(wave)

signal_without_background = flux_emission - cont
signal = signal_without_background
signal = np.nan_to_num(signal)
signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))


data = np.array(list(zip(wave, signal)))
np.savetxt(restore_path+str(name)+'.txt', data)

plt.plot(wave, flux_raw, label='Original signal')
plt.plot(wave, cont, label='Cont spec')
# plt.plot(wave, signal)


# times_to_mark = [6.220, 7.626, 8.60]
# colors = ['red', 'green', 'blue']
# for t,c in zip(times_to_mark,colors):
#     plt.axvline(x=t, color=c, label=str(t), linestyle='--')

plt.legend()
plt.xlabel('wavelength(μm)')
plt.ylabel('Intensity')
plt.title(str(name))
plt.savefig(compare+str(name)+'.png')
# plt.savefig(pic_path+str(name)+'.png')
plt.show()
