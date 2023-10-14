import torchaudio.functional as F
import torchaudio
import pandas as pd
import numpy as np
import argparse
import random
import joblib
import torch

from .predict import *
from .calculation import *
from .sound import *
from .extractData import *
from .sound_spearation import sound_separation

train_dict = {"hydrogen": 44, "newgen": 122}
config_dict = {"detector_path": "source/result/trained_model/best_model.pt",
               "regressor": "source/result/trained_model/linear_regression.pkl",
               "clean_sound_path": "source/result/sample_sound/generated_clean_sound.wav",
               "sample_tdms":"./data/221109_hydrogen/S206/test_2.tdms"}

def restrict_seed(seed):
    print("시드 고정: ", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(train_type, mode="lightweight"):
    restrict_seed(42)

    # 수소열차인지 차세대 열차인지 구분
    train_length = train_dict.get(train_type)
    print("열차의 종류는", train_type, "열차의 길이는", str(train_length)+"m 입니다.")

    # extractSoundAndDataFromTDMS 메서드로 LPData와 RawDate의 97번 채널 데이터를 추출합니다. 
    # LPData는 편의상 wav파일로 별도 저장합니다. 
    LPData_tdms, RawData_tdms = extractSoundAndDataFromTDMS(config_dict['sample_tdms'], config_dict['clean_sound_path'])
    assert (RawData_tdms is not None) | (LPData_tdms is not None), "tdms파일 추출 오류"

    # real world appliacation을 위해 깨끗한 데이터에 다른 소리를 병합합니다. 
    waveform, sample_rate = torchaudio.load( config_dict['clean_sound_path'] )
    audio_sample = F.resample(waveform[0], sample_rate, 25600, lowpass_filter_width=6)
    audio_noisy_sample = mixOtherSound(audio_sample, mix_rate=0.4)

    plot(audio_sample, audio_noisy_sample)

    mel_converter = torchaudio.transforms.MelSpectrogram(sample_rate=25600, n_mels=80)
    db_converter = torchaudio.transforms.AmplitudeToDB()

    feature = db_converter(mel_converter(audio_sample))

    isHorn = predict_anomaly(feature, config_dict['detector_path'], "cpu", mode=mode)

    if isHorn is True:
        if mode == "high_accuracy":
            print("음원 분리 실행")

            torchaudio.save("source/result/sample_sound/horn_detected_noisy.wav", audio_noisy_sample.unsqueeze(0), 25600)
            sound1_path, sound2_path = sound_separation("source/result/sample_sound/horn_detected_noisy.wav")

            print("음원 분리 완료")
            waveform_separated, sample_rate = torchaudio.load( sound1_path )
            audio_sample_separated = F.resample(waveform_separated[0], sample_rate, 25600, lowpass_filter_width=6)
            feature = db_converter(mel_converter(audio_sample_separated))

        # 위치 추정을 위해 horn의 peak시간과 열차가 감지된 시간을 체크합니다. (channel 97트리거 이용)
        start_sec, end_sec, peak_sec = extractInformationFromNumpy(RawData_tdms, feature)
        velocity = calculateAverageVelocity(train_length, start_sec, end_sec)

        # 열차의 평균속도와 peak데이터를 이용하여 선형회귀 추정을 합니다. 
        predict_location(peak_sec,velocity)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train anomaly detection')
    parser.add_argument('--traintype', type=str, default='hydrogen', choices=['hydrogen', 'newgen'])
    parser.add_argument('--mode', type=str, default='lightweight', choices=['lightweight', 'high_accuracy'])
    args = parser.parse_args()
    main(args.traintype, mode=args.mode)

# python -m source.main.inference.detect