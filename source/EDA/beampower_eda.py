from ..toolkit.plotClass import PlotManager
from ..toolkit.videoClass import VideoManager
from ..toolkit.tdmsClass import TdmsClass
from ..toolkit.jsonClass import JsonClass
from ..toolkit.utils import Uility

from tqdm import tqdm
from nptdms import TdmsFile
import math
import copy
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def main(paths):
    prefix = "./data/"

    filepath, json_filepath, filename = paths
    filepath = prefix + filepath
    json_filepath = prefix + json_filepath

    tdmsclass = TdmsClass(filepath)
    filled_list = Uility.fillingForNumeric( copy.copy(tdmsclass.file_list) ) #중간에 비어있거나 txt파일인 경우 None으로 채워넣기
    tdms_datas = tdmsclass.loadTdmsData( tdmsclass.file_list )
    tdms_datas = tdmsclass.getChannelData(tdms_datas, "Beampower", "Beampower")

    jsonclass = JsonClass(json_filepath)
    json_datas = jsonclass.loadJsonData( jsonclass.file_list )

    videomanager = VideoManager(size=(40, 30))

    count = 0
    for i in tqdm(range(len(filled_list))):
        if (filled_list[i] is None) | (tdms_datas[i] is None) | (json_datas[i] is None):
            tdms_datas.insert(i, None)

            count = count + 1
            continue

        tdms_value = tdms_datas[i]
        json_value = json_datas[i]

        title = "Num_" + str(i) + "_Horn " + str( json_value.get('Horn') ) + "_" + \
            str( json_value.get('Car_num') ) + "_" + str( json_value.get('Position') )
        filename = "source/result/" + title + "_beampower.mp4"

        if isinstance(tdms_value, np.ndarray):
            if count == 0:
                #save as video
                videomanager.reshape(tdms_value, (-1, 40, 30)).saveVideo(filename=filename)
        count = count + 1

    print("done")

if __name__ == '__main__':
    path_list = ["221108_nextgen/S206/*.tdms", "221109_hydrogen/S206/*.tdms"]
    json_path = ["train_json/221108_차세대전동차/*.json", "train_json/221109_수소열차/*.json"]
    name_list = ["221108_nextgen_S206", "221109_hydrogen_S206"]
    for path in zip(path_list, json_path, name_list):
        print(path)
        main(path)

#파일 실행 명령: python -m source.EDA.beampower_eda
#du -hsx * | sort -rh | head -n 10