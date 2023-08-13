from ..toolkit.plotClass import PlotManager
from ..toolkit.tdmsClass import TdmsClass

from tqdm import tqdm
from nptdms import TdmsFile
import math
def main():
    
    prefix = "./data/"
    filepath = prefix + "221108_nextgen/S206/*.tdms"
    tdmsclass = TdmsClass(filepath)

    tdms_datas = tdmsclass.loadTdmsData( tdmsclass.file_list )
    tdms_datas = tdmsclass.getChannelData(tdms_datas, "LPData", "Channel")

    weight, height = tdmsclass.get_list_to_matrix_size(tdms_datas)

    plotmanager = PlotManager(row=weight, col=height, type='plot_numpy')
    plot_rawaudio_list = []
    
    col_count = 0
    row_count = 0
    for i in tqdm(range(len(tdms_datas))):

        tdms_value = tdms_datas[i]
        row = {"content": tdms_value, "position": [col_count, row_count], "title":"TDMS"}
        plot_rawaudio_list.append( row )

        if col_count < weight:
            col_count = col_count+1
        if col_count == weight:
            row_count = row_count+1
            col_count = 0
    plotmanager.drawPlot(plot_rawaudio_list, save_as_file="source/result/LPData_values4.png")


if __name__ == '__main__':
    main()

#파일 실행 명령: python -m source.EDA.tdmsfile_eda