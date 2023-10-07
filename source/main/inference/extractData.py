from source.toolkit.tdmsClass import TdmsClass
from .sound import *
import numpy as np
from .calculation import calculatePeakPoint

def extractSoundAndDataFromTDMS(tdms_path, clean_path):
    print("extract data from tdms file")
    try:
        tdmsclass = TdmsClass(tdms_path)
        
        tdms_datas_lp = tdmsclass.loadTdmsData( tdmsclass.file_list )
        tdms_datas_lp = tdmsclass.getChannelData(tdms_datas_lp, "LPData", "Channel")
        tdms_datas_lp = tdms_datas_lp[0]
        saveToWav(tdms_datas_lp, clean_path)

        tdms_datas_raw = tdmsclass.loadTdmsData( tdmsclass.file_list )
        tdms_datas_raw = tdmsclass.getChannelData(tdms_datas_raw, "RawData", "Channel97")
        tdms_datas_raw = tdms_datas_raw[0]
        return tdms_datas_lp, tdms_datas_raw
    
    except Exception as e:
        print("Error occured from extractSoundAndDataFromTDMS: ", e)
        return None

def extractInformationFromNumpy(RawData, LPData_feature):
    total_time = len(RawData) / 25600
    sampling = list(RawData[::int(25600/5)])

    # peak = float( np.argmax(LPData) / 25600) if len(LPData)>0 else -1
    peak = calculatePeakPoint(total_time, LPData_feature)

    start = sampling.index(1.0) if 1.0 in sampling else -1
    sampling.reverse()
    end = sampling.index(1.0) if 1.0 in sampling else -1
    end_point = int( len(sampling) - end )
    return  float(start/5), float(end_point/5), peak