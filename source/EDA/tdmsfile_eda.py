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
    filepath = prefix + "221108_nextgen/S206/*.tdms"
    json_filepath = prefix + "train_json/221108_차세대전동차/*.json"

    tdmsclass = TdmsClass(filepath)
    filled_list = Uility.fillingForNumeric( copy.copy(tdmsclass.file_list) ) #중간에 비어있거나 txt파일인 경우 None으로 채워넣기
    tdms_datas = tdmsclass.loadTdmsData( tdmsclass.file_list )
    tdms_datas = tdmsclass.getChannelData(tdms_datas, "LPData", "Channel")

    weight, height = tdmsclass.get_list_to_matrix_size(tdms_datas)

    jsonclass = JsonClass(json_filepath)
    json_datas = jsonclass.loadJsonData( jsonclass.file_list )

    plotmanager = PlotManager(row=weight, col=height, type='plot_numpy')
    plot_rawaudio_list = []

    col_count = 0
    row_count = 0
    for i in tqdm(range(len(filled_list))):
        if filled_list[i] is None:
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
    
    plotmanager.drawPlot(plot_rawaudio_list, 
    save_as_file="source/result/221108_nextgen_S206_LPData_values_with_json.png")

if __name__ == '__main__':
    main()

#파일 실행 명령: python -m source.EDA.tdmsfile_eda