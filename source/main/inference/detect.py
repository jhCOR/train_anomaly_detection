import torchaudio
import torchaudio.functional as F
import torch
from torchvision.models import resnet18
import torch.nn as nn
import joblib
import copy
from source.toolkit.tdmsClass import TdmsClass
from source.toolkit.utils import Uility
import pandas as pd
import numpy as np
import argparse

def number_of_correct(pred, target):
    # count number of correct predictions
    return pred.squeeze().eq(target).sum().item()

def get_likely_index(tensor):
    # find most likely label index for each element in the batch
    return tensor.argmax(dim=-1)

def prepare_model():
    resnet_model = resnet18( weights='IMAGENET1K_V1' )
    resnet_model.fc = nn.Linear(512,2)

    for m in resnet_model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    return resnet_model

def predict_anomaly(audio_sample, model_path, device):
    print("start inference")
    #yes가 1임
    mel_converter = torchaudio.transforms.MelSpectrogram(sample_rate=25600, n_mels=80)
    db_converter = torchaudio.transforms.AmplitudeToDB()

    feature = db_converter(mel_converter(audio_sample))

    model = prepare_model()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.to(device)
    model.eval()

    feature = feature.unsqueeze(0)
    expanded_data = torch.stack([feature]*3, dim=1)
    expanded_data = expanded_data.to(device)
    output = model(expanded_data)
    pred = get_likely_index(output)

    print( output )
    print( "이상치 감지" if int(pred[0]) == 1 else "이상치 불검출")

def predict_location(peak, average_velocity):
    data = pd.DataFrame({"peak": [peak], "average_velocity(km)":[average_velocity]})
    regressor = joblib.load("source/result/trained_model/linear_regression.pkl")
    prediction = regressor.predict(data)
    print( "위치 추정: ", str(prediction[0]) + "m" )

def saveToWav(tdms_value):
    if len(tdms_value)>0:
        print("save to wav file")
        try:
            tensor_data = torch.Tensor(tdms_value).unsqueeze(0)
            path = f"source/result/sample_sound/generated_noisy_sound.wav"
            torchaudio.save(path, tensor_data, 25600)

        except Exception as e:
            print("error:", e)
        print("successfully save tdms(LPData) to wav.")
    else:
        print("too short tdms file")

def extractSoundAndDataFromTDMS(tdms_path):
    print("extract data from tdms file")
    try:
        
        tdmsclass = TdmsClass(tdms_path)
        
        tdms_datas_lp = tdmsclass.loadTdmsData( tdmsclass.file_list )
        tdms_datas_lp = tdmsclass.getChannelData(tdms_datas_lp, "LPData", "Channel")
        tdms_datas_lp = tdms_datas_lp[0]
        saveToWav(tdms_datas_lp)

        tdms_datas_raw = tdmsclass.loadTdmsData( tdmsclass.file_list )
        tdms_datas_raw = tdmsclass.getChannelData(tdms_datas_raw, "RawData", "Channel97")
        tdms_datas_raw = tdms_datas_raw[0]
        return tdms_datas_lp, tdms_datas_raw
    
    except Exception as e:
        print("Error occured from extractSoundAndDataFromTDMS: ", e)
        return None

def extractInformationFromTDFS(RawData, LPData):
    sampling = list(RawData[::int(25600/5)])
    peak = float( np.argmax(LPData) / 25600) if len(LPData)>0 else -1
    start = sampling.index(1.0) if 1.0 in sampling else -1
    sampling.reverse()
    end = sampling.index(1.0) if 1.0 in sampling else -1
    end_point = int( len(sampling) - end )
    return  float(start/5), float(end_point/5), peak

def calculateAverageVelocity(train_length, start, end):
    average_velocity = ( train_length * 0.001)  / ( ( end - start ) / 3600)
    return average_velocity
train_dict = {"hydrogen": 44, "newgen": 122}

def main(train_type):
    train_length = train_dict.get(train_type)
    filePath = "./data/221109_hydrogen/S206/test_2.tdms"
    LPData_tdms, RawData_tdms = extractSoundAndDataFromTDMS(filePath)
    if (RawData_tdms is None) | (LPData_tdms is None):
        print("something is wrong with extractSoundAndDataFromTDMS")

    start_sec, end_sec, peak_sec = extractInformationFromTDFS(RawData_tdms, LPData_tdms)
    velocity = calculateAverageVelocity(train_length, start_sec, end_sec)
    waveform, sample_rate = torchaudio.load( "source/result/sample_sound/generated_noisy_sound.wav")
    audio_sample = F.resample(waveform[0], sample_rate, 25600, lowpass_filter_width=6)
    modelPath = "source/result/trained_model/best_model.pt"
    predict_anomaly(audio_sample, modelPath, "cpu")
    predict_location(peak_sec,velocity)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train anomaly detection')
    parser.add_argument('--traintype', type=str, default='hydrogen', choices=['hydrogen', 'newgen'])

    args = parser.parse_args()
    main(args.traintype)

# python -m source.main.inference.detect