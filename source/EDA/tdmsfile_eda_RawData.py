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
import pandas as pd

sampling_rate = 22050

def inspect_file(path):
    print("-" * 10)
    print("Source:", path)
    print("-" * 10)
    print(f" - File size: {os.path.getsize(path)} bytes")
    print(f" - {torchaudio.info(path)}")
    print()

def extractData(df, json_data, path, start, end, peak_point):
    title=json_data.get('title_s206')
    Horn = json_data.get('Horn')
    Position = json_data.get('Position')

    S206_position = json_data.get('S206_position')

    Car_type = json_data.get('Train')
    Length = json_data.get('Length')
    Car_num = json_data.get('Car_num')
    df2 = pd.DataFrame.from_dict([{'title': title, 'Horn' : Horn, 'Position' : Position, 
                                   'S206_position' : S206_position,'Car_type':Car_type, 
                                   'Length':Length,  'Car_num':Car_num, 'path': path, 
                                   "channel97_start":start, "channel97_end":end,
                                   "peak": peak_point}])
    new_df = pd.concat([df,df2])

    return new_df

def EDA_tdms(paths, meta_dataframe):
    
    prefix = "./data/"
    filepath, json_filepath, filename = paths
    filepath = prefix + filepath
    json_filepath = prefix + json_filepath
    #group = property.get('group')
    #channel = property.get('channel')

    tdmsclass = TdmsClass(filepath)
    filled_list = Uility.fillingForNumeric( copy.copy(tdmsclass.file_list) ) #중간에 비어있거나 txt파일인 경우 None으로 채워넣기
    tdms_datas = tdmsclass.loadTdmsData( tdmsclass.file_list )
    tdms_datas = tdmsclass.getChannelData(tdms_datas, "RawData", "Channel97")
    tdms_datas_2 = tdmsclass.loadTdmsData( tdmsclass.file_list )
    tdms_datas_2 = tdmsclass.getChannelData(tdms_datas_2, "LPData", "Channel")

    weight, height = tdmsclass.get_list_to_matrix_size(tdms_datas)

    jsonclass = JsonClass(json_filepath)
    json_datas = jsonclass.loadJsonData( jsonclass.file_list )

    plotmanager = PlotManager(row=weight, col=height, type='plot_numpy')
    plot_rawaudio_list = []

    col_count = 0
    row_count = 0

    print(len(filled_list))
    for i in tqdm(range(len(filled_list))):
        if (filled_list[i] is None) | (tdms_datas[i] is None) | (json_datas[i] is None) | (tdms_datas_2[i] is None):
            plot_rawaudio_list.append( None )
            tdms_datas.insert(i, None)
            tdms_datas_2.insert(i, None)

            if col_count < weight:
                col_count = col_count+1
            if col_count == weight:
                row_count = row_count+1
                col_count = 0
            continue

        tdms_value = tdms_datas[i]
        tdms_value_2 = tdms_datas_2[i]
        json_value = json_datas[i]

        title = "Num_" + str(i) + "_Horn_" + str( json_value.get('Horn') ) + "_" + \
            str( json_value.get('Car_num') ) + "_" + str( json_value.get('Position') )
        row = {"content": tdms_value[::25600], "position": [col_count, row_count], "title":title}

        plot_rawaudio_list.append( row )

        path = f"{filename}_{title}.wav"
        
        sampling_rate = 5
        sampling = list(tdms_value[::int(25600/5)])
        peak = float( np.argmax(tdms_value_2) / 25600) if len(tdms_value_2)>0 else -1
        start = sampling.index(1.0) if 1.0 in sampling else -1
        sampling.reverse()
        end = sampling.index(1.0) if 1.0 in sampling else -1
        end_point = int( len(sampling) - end )
        meta_dataframe = extractData(meta_dataframe, json_value, path, float(start/5), float(end_point/5), peak)

        # if len(tdms_value)>1:
        #     try:
        #         tensor_data = torch.Tensor(tdms_value).unsqueeze(0)
        #         path = f"source/result/wav/{filename}_{title}.wav"
        #         torchaudio.save(path, tensor_data, sampling_rate)
        #         #inspect_file(path)
        #     except Exception as e:
        #         print("error:", e)
        #print(torch.Tensor(tdms_value).shape)
        if col_count < weight:
            col_count = col_count+1
        if col_count == weight:
            row_count = row_count+1
            col_count = 0
    filename = "source/result/" + filename +"_RawData_values_with_json.png"

    plotmanager.drawPlot(plot_rawaudio_list, save_as_file=filename)
    
    print("done")
    return meta_dataframe


path_list =  ["221102_hydrogen", "221103_nextgen", "221104_nextgen", "221107_nextgen",
               "221108_nextgen", "221109_hydrogen", "221110_nextgen"]
json_path = ["221102_수소열차", "221103_차세대전동차", "221104_차세대전동차", "221107_차세대전동차",
              "221108_차세대전동차", "221109_수소열차", "221110_차세대전동차"]

if __name__ == '__main__':
    full_path_list = [str(i) + "/S206/*.tdms" for i in path_list]
    full_json_path = ["train_json/" + str(j) + "/*.json" for j in json_path]
    full_name_list = [str(i) + "_S206" for i in path_list]
    #property_list = [{"group": "LPData", "channel": "Channel"}]

    dataframe = pd.DataFrame(columns=['Horn','Position', 'S206_position',
                        'Car_type', 'Length',  'Car_num'])
    count = 0
    for path in zip(full_path_list, full_json_path, full_name_list):
        print(count, "번째 -> ", path)
        dataframe = EDA_tdms(path, dataframe)
        count = count + 1
    dataframe.to_csv("source/result/metadata.csv")
#파일 실행 명령: python -m source.EDA.tdmsfile_eda_RawData