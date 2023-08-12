from .utils import Uility
import glob
import numpy as np

class TdmsClass(Uility):
    def __init__(self, file_path):
        files = glob.glob(file_path)
        self.file_list = self.loadData(files)

    def loadData(self, filepaths):
        sorted_list = self.sort_filenames_by_number(filepaths, criteria='test_')
        return sorted_list
    
    def reFormData(self):
        sorted_numpy_list = np.array(self.file_list)
        size = self.get_square_range(len(sorted_numpy_list))
        padded_list = self.paddingList(size*size, sorted_numpy_list)
        reshaped_list = np.reshape(padded_list, (size, -1))
        return reshaped_list

    def getData(self):
        return 0
