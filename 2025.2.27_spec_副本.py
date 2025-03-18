import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import stft
import os

# read eeg data

# 读取txt数据
def readData(filefold, filename):
    dataList = []
    filepath = filefold + "/" + filename
    with open(filepath, 'r') as file:
        lines = file.readlines()
    for line in lines:
        fields = line.strip('\n') #去除换行符
        dataList.append(fields)
    dataList = np.array(dataList)
    dataList = dataList.astype(float)
    return dataList

# 计算spec图像
def eeg_stft_spectrogram(filepath, filename, sample_rate, nperseg, start, end):
    eeg_data = []
    filepath = filepath + "/" + filename
    with open(filepath, 'r') as file:
        lines = file.readlines()
    for line in lines:
        fields = line.strip('\n') #去除换行符
        eeg_data.append(fields)
    
    eeg_data = eeg_data[start:end]
    eeg_data = np.array(eeg_data)
    eeg_data = eeg_data.astype(float)
    
    # Compute the Short-Time Fourier Transform (STFT)
    f, t, Zxx = stft(eeg_data, fs=sample_rate, nperseg=nperseg)
    
    # Compute the magnitude (spectrogram)
    spectrogram = np.abs(Zxx)

    return f, t, spectrogram

# save the spec image
def save_spec(participant, channel, file_num, f, t, spectrogram):
    # Plot the spectrogram
    plt.figure(figsize=(5, 5))  # Set the figure size
    plt.pcolormesh(t, f, np.log10(spectrogram +1e-12), shading='gouraud', cmap='jet')
    plt.tight_layout()  # Adjust layout for better appearance
    plt.xticks([])
    plt.yticks([])
    
    filepath = 'spec/不同人/'+ participant + '/'+ channel
    filename = filepath + '/' + str(file_num) + '.png'
    os.makedirs(filepath, exist_ok=True)
    plt.savefig(filename, bbox_inches='tight')


file_path = "64_eeg_channel_data/responder"
channel_name = "FCz.txt"  #channel name
file_names = os.listdir(file_path)

gap = 3000
start = 10000
end = start + gap

# HC group(0-23)
numbers = list(range(24))

# 转换为固定三字符宽度的字符串
filenames = [f"{num:03}" for num in numbers]
file_num = 0

for file_name in filenames:
    filename = file_name + '_responder'
    filepath = file_path + '/' + filename
    try:
        data = readData(filepath,channel_name)
        gap = 3000
        start = 10000
        end = start + gap
    
        participant = 'HC'
        channel = 'FCz'
    
        for i in range(0,20):
            f, t, spec = eeg_stft_spectrogram(filepath, channel_name, 256, 256, start, end)
            save_spec(participant, channel, file_num, f, t, spec)
            start = end
            end = start + gap
            file_num += 1
            #print("第%d次已完成" % file_num)
    except Exception as e:
        print(f"Error loading BrainVision file: {e}")

    print(filename + "已完成")

# P group(24-42)
numbers = list(range(42))
filenames = [f"{num:03}" for num in numbers]
filenames = filenames[25:]
file_num = 0

for file_name in filenames:
    filename = file_name + '_responder'
    filepath = file_path + '/' + filename
    try:
        data = readData(filepath,channel_name)
        gap = 3000
        start = 10000
        end = start + gap
    
        participant = 'P'
        channel = 'FCz'
    
        for i in range(0,20):
            f, t, spec = eeg_stft_spectrogram(filepath, channel_name, 256, 256, start, end)
            save_spec(participant, channel, file_num, f, t, spec)
            start = end
            end = start + gap
            file_num += 1
            #print("第%d次已完成" % file_num)
    except Exception as e:
        print(f"Error loading BrainVision file: {e}")