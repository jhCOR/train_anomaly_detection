from .objectHelper import ObjectHelper
import re
import math
import numpy as np

type_dict = {'str':str, "int":int}

class Uility():
    @classmethod
    def get_list_to_matrix_position(cls):
        return 0

    @classmethod
    def get_list_to_matrix_size(cls, lists):
        
        weight = cls.get_square_range(len(lists))
        height = math.ceil( len(lists)/weight)
        return weight, height

    @classmethod
    def get_square_range(cls, length):
        square_root = math.ceil(length ** 0.5)
        return square_root

    def reFormShape(self, lists, row=None, col=None):
        numpy_list = np.array(lists)
        self.size = self.get_square_range(len(numpy_list))
        padded_list = self.paddingList(self.size*self.size, numpy_list)

        row_length = self.size if row is None else row
        col_length = -1 if col is None else col

        reshaped_list = np.reshape(padded_list, (row_length, col_length))
        return reshaped_list

    def paddingList(self, amount, lists):
        if isinstance(lists, np.ndarray):
            lists = lists.tolist()
            for i in range(amount-len(lists)):
                lists.append(lists[-1])
            print(lists)
            lists = np.array(lists)
        else:
            for i in range(amount-len(lists)):
                lists.append(lists[-1])
        return lists
    
    def extract_number(self, filename):
        part = r'_random_(\d{1,2})'
        target = re.sub('_random_', self.criteria, part)
        match = re.search(target, filename)

        if match:
            return int(match.group()[len(self.criteria):])
        else: 
            target = re.sub('_random_', self.sub_criteria, part)
            match = re.search(target, filename)
            return int(match.group()[len(self.sub_criteria):])

    def sort_filenames_by_number(self, filenames, criteria="_out_", sub_criteria="_"):
        self.criteria = criteria
        self.sub_criteria = sub_criteria
        return sorted(filenames, key=self.extract_number)

    @classmethod
    def extractSubString(cls, data_list, criteria, return_type='int'):
        data_list = data_list if ObjectHelper.isIterable(data_list) else [data_list]

        part = r'_random_(\d{1,2})'
        target = re.sub('_random_', criteria, part)

        result_list = []
        for data in data_list:
            match = re.search(target, data)
            type_maker = type_dict.get(return_type)

            if match:
                result_list.append( type_maker( match.group()[len(criteria):] ) )
        return result_list
    
    @classmethod
    def fillingForNumeric(cls, data_list, fill_with=None):
        #list must be sorted!!

        data_list = data_list if ObjectHelper.isIterable(data_list) else [data_list]
        
        number_list = cls.extractSubString(data_list, "_", "int")

        for i in range(int(number_list[-1])):
            if i+1 in number_list:
                continue
            else:
                number_list.append(i+1)
        number_list = sorted(number_list)

        for i in range(len(number_list)):
            find = False
            for index in range(len(data_list)):
                target = '_{0:02d}'.format(number_list[i])

                if data_list[index].find(str(target)) > -1:
                    find = True
                    break
            if find is False:
                print(number_list[i])
        data_list.insert(number_list[i]-2, None)
        return data_list

        



    


