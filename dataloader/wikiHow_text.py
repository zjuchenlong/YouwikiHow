"""
https://github.com/mahnazkoupaee/WikiHow-Dataset

download from:  (WikihowAll.csv) https://ucsb.box.com/s/ap23l8gafpezf4tq3wapr6u8241zz358
                (WikihowSep.csv) https://ucsb.box.com/s/7yq601ijl1lzvlfu4rjdbbxforzd2oag
"""

import os
import pandas as pd
from pathlib import Path


class wikiHowTextDataset():

    def __init__(self):

        Dataset_path = Path('../Datasets')
        if not os.path.exists(Dataset_path):
            print(f"{Dataset_path} is not exist!")
            exit(0)

        ## wikiHow dataset
        wikiHow_path = Dataset_path / 'wikiHow/WikiHow-Dataset'
        wikihowAll = pd.read_csv(wikiHow_path / 'wikihowAll.csv')
        wikihowSep = pd.read_csv(wikiHow_path / 'wikihowSep.csv')
        self.wikihowAll = wikihowAll.astype(str)
        self.wikihowSep = wikihowSep.astype(str)
        self.wikiHow_tasks = self._filter_nan_tasks(wikihowAll['title'].to_list())

    def _filter_nan_tasks(self, tasks):
        return_tasks = list()
        for task_i in tasks:
            if type(task_i) == str:
                return_tasks.append(task_i)
        return return_tasks
