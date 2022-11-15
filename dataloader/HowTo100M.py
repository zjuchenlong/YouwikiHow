"""
https://www.di.ens.fr/willow/research/howto100m/

All-in-One zip

"""

import os
import pandas as pd
from pathlib import Path

class HowTo100MDataset():

    def __init__(self):
        Dataset_path = Path('../Datasets')
        if not os.path.exists(Dataset_path):
            print(f"{Dataset_path} is not exist!")
            exit(0)
        howto_path = Dataset_path / 'HowTo100M'
        howto_tasks_file = pd.read_csv(howto_path / 'task_ids.csv', error_bad_lines=False, header=None)
        howto_tasks_file = howto_tasks_file[0].to_list()
        self.howto_videos_file = pd.read_csv(howto_path / 'HowTo100M_v1.csv')

        self.howto_tasks, self.howto_id2task, self.howto_task2id = self.howto_task_preprocessing(howto_tasks_file)
        self.howto_task_id = self.howto_videos_file.task_id

    def howto_task_preprocessing(self, task_list):

        task_name_list = list()
        task_id2name = dict()
        task_name2id = dict()
        for t in task_list:
            task_id, task_name = t.split('\t')
            task_id = int(task_id)
            task_name_list.append(task_name)
            task_id2name[task_id] = task_name
            task_name2id[task_name] = task_id

        return task_name_list, task_id2name, task_name2id


    def get_videos_by_task_id(self, task_id):
        return self.howto_videos_file.loc[self.howto_task_id == task_id]