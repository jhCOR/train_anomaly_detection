from ..toolkit.plotClass import PlotManager
from ..toolkit.tdmsClass import TdmsClass
from ..toolkit.jsonClass import JsonClass
from ..toolkit.utils import Uility

from tqdm import tqdm
import numpy as np
import wave, math
import math
import copy
import tempfile
import torchaudio
import os
import torch 
from pathlib import Path

sampling_rate = 22050

##미완성 코드

def sound_merger(paths, exp=".wav"):
    import os
    path_dir = 'source/result/wav/'
    file_list = os.listdir(path_dir)
    print(file_list)

    waveform, sample_rate = torchaudio.load(path_dir+file_list[0])
    front = int( (len(waveform) - len(raw_1)) / 4 )
    back = (len(waveform) - len(raw_1)) - front
    print(front, back)
    raw_1 = np.pad(raw_1, (front, back) , 'constant', constant_values=0)
    merged = waveform + raw_1*10
    torchaudio.save("/source/result/merged_sound/test_merge_sound.wav", torch.Tensor(merged).unsqueeze(0), sample_rate)
    return 0

if __name__ == '__main__':
    path="/source/result/wav/221108_nextgen_S206_Num_0_Horn_Yes_6_61.wav"
    sound_merger(path)

#파일 실행 명령: python -m source.EDA.sound_merging