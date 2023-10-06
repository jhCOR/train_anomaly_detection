import torch
import torchaudio
import torch.nn as nn
import pandas as pd
import random
import torchaudio.functional as F
import glob

def saveToWav(tdms_value, path):
    if len(tdms_value)>0:
        print("tdms의 LPData를 wav파일로 저장합니다. ")
        try:
            tensor_data = torch.Tensor(tdms_value).unsqueeze(0)
            torchaudio.save(path, tensor_data, 25600)

        except Exception as e:
            print("error:", e)
        print("successfully save tdms(LPData) to wav.")
    else:
        print("too short tdms file")

def get_other_sound():
    print("병합할 다른 음원을 가져옵니다. ")
    othersourcePath = "source/result/MTAT_data/0/*.mp3"
    path_list =glob.glob(othersourcePath)

    path = random.randrange(1,len(path_list))

    waveform, sample_rate = torchaudio.load( path_list[path] )
    audio_sample = F.resample(waveform[0], sample_rate, 25600, lowpass_filter_width=6)
    return audio_sample

def minmaxScaler(data):
    # 정규화를 수행합니다. 
    # horn이 일반적으로 데시벨이 크지만 보다더 robust한 모델 학습/추론을 위해 정규화하여
    # 페널티를 부여합니다. 
    data_min, data_max = data.min(), data.max()
    new_lower, new_upper = -1, 1
    scaled_data = (data - data_min)/(data_max - data_min)*(new_upper - new_lower) + new_lower
    return scaled_data

def mixOtherSound(audio_sample, mix_rate=1.0):
    # 시험 데이터를 실 데이터로 전이시키는 함수
    other_sound = get_other_sound()
    if len(other_sound) > len(audio_sample):
        other_sound = other_sound[:len(audio_sample)]
    else:
        other_sound = other_sound[:len(audio_sample)]
        front = int( (len(audio_sample) - len(other_sound)) / 4 ) * 3
        back = (len(audio_sample) - len(other_sound)) - front
        other_sound = nn.functional.pad(other_sound, (front, back) , 'constant', value=0)

    scaled_audio_data = minmaxScaler(audio_sample)
    scaled_other_data = minmaxScaler(other_sound)

    audio_sample = scaled_audio_data + (scaled_other_data*mix_rate)

    return audio_sample