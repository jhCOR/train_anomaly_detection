from collections.abc import Iterable
import re
import math
import numpy as np

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

class ObjectHelper():
    @classmethod
    def is_NestedIterable(cls, target):
        if cls.isIterable(target) and all(cls.isIterable(target) for sublist in target):
            return True
        else:
            return False

    @classmethod
    def isIterable(self, target):
        return True if isinstance(target, Iterable) else False

