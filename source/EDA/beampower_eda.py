from ..toolkit.plotClass import PlotManager
from ..toolkit.tdmsClass import TdmsClass
from ..toolkit.jsonClass import JsonClass
from ..toolkit.utils import Uility

from tqdm import tqdm
from nptdms import TdmsFile
import math
import copy

def main():
    prefix = "./data/"
    filepath, json_filepath, filename = paths
    filepath = prefix + filepath
    json_filepath = prefix + json_filepath
    # filepath = prefix + "221108_nextgen/S206/*.tdms"
    # json_filepath = prefix + "train_json/221108_차세대전동차/*.json"

    tdmsclass = TdmsClass(filepath)
    filled_list = Uility.fillingForNumeric( copy.copy(tdmsclass.file_list) ) #중간에 비어있거나 txt파일인 경우 None으로 채워넣기
    tdms_datas = tdmsclass.loadTdmsData( tdmsclass.file_list )
    tdms_datas = tdmsclass.getChannelData(tdms_datas, "Beampower", "Beampower")

    weight, height = tdmsclass.get_list_to_matrix_size(tdms_datas)

    jsonclass = JsonClass(json_filepath)
    json_datas = jsonclass.loadJsonData( jsonclass.file_list )

    plotmanager = PlotManager(row=weight, col=height, type='plot_numpy')
    plot_rawaudio_list = []

if __name__ == '__main__':
    main()

#파일 실행 명령: python -m source.EDA.beampower_eda.py