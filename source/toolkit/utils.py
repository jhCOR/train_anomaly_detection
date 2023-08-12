from collections.abc import Iterable
import re
import math
import numpy as np

class Uility():
    @classmethod
    def get_list_to_matrix_position(cls):
        return 0

    @classmethod
    def get_list_to_matrix_size(cls):
        return 0

    @classmethod
    def get_square_range(cls, length):
        square_root = math.ceil(length ** 0.5)
        return square_root

    def paddingList(self, amount, lists):
        if isinstance(lists, np.ndarray):
            lists = lists.tolist()
            for i in range(amount-len(lists)):
                lists.append('')
            lists = np.array(lists)
        else:
            for i in range(amount-len(lists)):
                lists.append([])
        
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

