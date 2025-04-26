'''
Concrete ResultModule class for a specific experiment ResultModule output
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.result import result
import pickle


class Result_Saver(result):
    data = None
    fold_count = None
    result_destination_folder_path = None
    result_destination_file_name = None

    def __init__(self, sName=None, sDescription=None, file_suffix=''):
        super().__init__(sName, sDescription)
        self.file_suffix = file_suffix

    def save(self):
        print('saving results...')
        f = open(self.result_destination_folder_path + self.result_destination_file_name + '_' + str(self.fold_count) + "_" + self.file_suffix, 'wb')
        pickle.dump(self.data, f)
        f.close()