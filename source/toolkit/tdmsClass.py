from .utils import Uility, ObjectHelper
import glob
import numpy as np
from nptdms import TdmsFile
from tqdm import tqdm

# tdms를 object화 하기
class TdmsObjectClass():
    def __init__(self, data):
        self.data = data
    def __call__():
        return 0
    
    def getChannelData(self):
        return 0
    def checkTdmsObject(self, data):
        return isinstance(data, self)
    def checkTdms(self, data):
        return isinstance(data, TdmsFile)


class TdmsClass(Uility):
    def __init__(self, file_path):
        super().__init__()
        files = glob.glob(file_path)
        self.file_list = self.loadPath(files)
        print(self)

    def loadPath(self, filepaths):
        sorted_list = self.sort_filenames_by_number(filepaths, criteria='test_')
        return sorted_list

    def loadTdmsData(self, filepaths):
        print("--load tdms data--")
        tdms_list = []
        for path in tqdm(filepaths):
            tdms_list.append( TdmsFile(path) )
        return tdms_list

    def getChannelData(self, tdms_datas, group, channel):
        if ObjectHelper.isIterable(tdms_datas) is False:
            tdms_datas = [tdms_datas]
        lists = [tdms[str(group)][str(channel)][:] for tdms in tdms_datas ]
        return lists

    def getinfo(self, tdms_file):
        group_list = tdms_file.groups()
        print("Groups in TDMS:", group_list)
        for group in group_list:
            print("Channels in Group:", tdms_file[group.name].channels())