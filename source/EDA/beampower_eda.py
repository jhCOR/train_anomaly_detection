from ..toolkit.plotClass import PlotManager
from ..toolkit.tdmsClass import TdmsClass
from ..toolkit.jsonClass import JsonClass
from ..toolkit.utils import Uility

from tqdm import tqdm
from nptdms import TdmsFile
import math
import copy

# tdmsclass.getChannelData(tdms_datas, "Beampower", "Beampower") 이 코드 제외하고 
# tdms_eda와 완전히 같은데 EDA다보니 편의상 분리하였습니다. 
def main(paths):
    prefix = "./data/"

    filepath, json_filepath, filename = paths
    filepath = prefix + filepath
    json_filepath = prefix + json_filepath

    tdmsclass = TdmsClass(filepath)
    filled_list = Uility.fillingForNumeric( copy.copy(tdmsclass.file_list) ) #중간에 비어있거나 txt파일인 경우 None으로 채워넣기
    tdms_datas = tdmsclass.loadTdmsData( tdmsclass.file_list )
    tdms_datas = tdmsclass.getChannelData(tdms_datas, "Beampower", "Beampower")

    weight, height = tdmsclass.get_list_to_matrix_size(tdms_datas)

    jsonclass = JsonClass(json_filepath)
    json_datas = jsonclass.loadJsonData( jsonclass.file_list )

    plotmanager = PlotManager(row=weight, col=height, type='plot_numpy')
    plot_rawaudio_list = []

    col_count = 0
    row_count = 0
    for i in tqdm(range(len(filled_list))):
        if (filled_list[i] is None) | (tdms_datas[i] is None) | (json_datas[i] is None):
            plot_rawaudio_list.append( None )
            tdms_datas.insert(i, None)

            if col_count < weight:
                col_count = col_count+1
            if col_count == weight:
                row_count = row_count+1
                col_count = 0
            continue

        tdms_value = tdms_datas[i]
        json_value = json_datas[i]

        title = "Num: " + str(i) + "-> Horn " + str( json_value.get('Horn') ) + "_" + \
            str( json_value.get('Car_num') ) + "_" + str( json_value.get('Position') )
        row = {"content": tdms_value, "position": [col_count, row_count], "title":title}
        plot_rawaudio_list.append( row )

        if col_count < weight:
            col_count = col_count+1
        if col_count == weight:
            row_count = row_count+1
            col_count = 0
    filename = "source/result/" + filename +"_Beampower_values_with_json.png"
    plotmanager.drawPlot(plot_rawaudio_list, save_as_file=filename)
    print("done")

if __name__ == '__main__':
    path_list = ["221108_nextgen/S206/*.tdms", "221109_hydrogen/S206/*.tdms"]
    json_path = ["train_json/221108_차세대전동차/*.json", "train_json/221109_수소열차/*.json"]
    name_list = ["221108_nextgen_S206", "221109_hydrogen_S206"]
    for path in zip(path_list, json_path, name_list):
        print(path)
        main(path)

#파일 실행 명령: python -m source.EDA.beampower_eda