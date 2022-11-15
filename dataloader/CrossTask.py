from pathlib import Path
import os

class CrossTaskDataset():

    def __init__(self):
        
        # CrossTask_dirname = Path('/dvmm-filer3a/users/chenlong/Datasets/CrossTask/crosstask_release')
        CrossTask_dirname = Path('../Datasets/CrossTask/crosstask_release')
        if not os.path.exists(CrossTask_dirname):
            print(f"{CrossTask_dirname} is not exist!")
            exit(0)
        tasks_primary_dirname = CrossTask_dirname / 'tasks_primary.txt'
        tasks_related_dirname = CrossTask_dirname / 'tasks_related.txt'
        tasks_videos_dirname = CrossTask_dirname / 'videos.csv'
        tasks_val_videos_dirname = CrossTask_dirname / 'videos_val.csv'
        step_annotation_dirname = CrossTask_dirname / 'annotations'

        self.tasks_primary_info = self.read_task_info(tasks_primary_dirname)
        self.tasks_related_info = self.read_task_info(tasks_related_dirname)
        self.videos = self.get_vids(tasks_videos_dirname)
        self.val_videos = self.get_vids(tasks_val_videos_dirname)
        self.all_videos = self.merge_val_videos(self.videos, self.val_videos)

        self.task2id, self.id2task = self.get_task_id_mapping()
        self.task2step = self.get_task_step_mapping()

        self.all_step_annotations = self.get_all_step_annnotations(step_annotation_dirname)


    def get_all_step_annnotations(self, path):

        all_step_annotations = dict()
        for (root, dirs, files) in os.walk(path):

            for file in files:
                f_name = file.split('.')[0]
                f_path = os.path.join(root, file)

                all_step_annotations[f_name] = list()
                with open(f_path,'r') as f:
                    step_annotation = f.readline().strip()
                    while step_annotation != '':
                        step_id, step_s, step_e = step_annotation.split(',')
                        all_step_annotations[f_name].append({'step_id': step_id,
                                                             'step_s': step_s,
                                                             'step_e': step_e})
                        step_annotation = f.readline().strip()
        return all_step_annotations


    def get_task_step_mapping(self):
        task2step = dict()
        all_tasks = self.get_all_tasks()
        for task_name in all_tasks:
            task_id = self.task2id[task_name]
            if task_id in self.tasks_primary_info['steps'].keys():
                task_step = self.tasks_primary_info['steps'][task_id]
                assert len(task_step) == self.tasks_primary_info['n_steps'][task_id]
                task2step[task_name] = task_step
            if task_id in self.tasks_related_info['steps'].keys():
                task_step = self.tasks_related_info['steps'][task_id]
                assert len(task_step) == self.tasks_related_info['n_steps'][task_id]
                task2step[task_name] = task_step
        return task2step


    def get_task_id_mapping(self):
        task2id = dict()
        id2task = dict()
        for task_id, task_name in self.tasks_primary_info['title'].items():
            task2id[task_name] = task_id
            id2task[task_id] = task_name
        for task_id, task_name in self.tasks_related_info['title'].items():
            task2id[task_name] = task_id
            id2task[task_id] = task_name
        return task2id, id2task


    def merge_val_videos(self, videos, val_videos):
        merged_videos = videos.copy()
        for k, v in val_videos.items():
            assert k in videos.keys()
            merged_videos[k].extend(val_videos[k])
        return merged_videos


    def get_all_tasks(self):
        all_tasks = list()
        all_tasks.extend(self.get_primary_tasks())
        all_tasks.extend(self.get_related_tasks())
        return all_tasks


    def get_primary_tasks(self):
        all_tasks = list()
        for k, v in self.tasks_primary_info['title'].items():
            all_tasks.append(v)
        return all_tasks


    def get_related_tasks(self):
        all_tasks = list()
        for k, v in self.tasks_related_info['title'].items():
            all_tasks.append(v)
        return all_tasks


    def read_task_info(self, path):
        titles = {}
        urls = {}
        n_steps = {}
        steps = {}
        with open(path,'r') as f:
            idx = f.readline()
            while idx != '':
                idx = idx.strip()
                titles[idx] = f.readline().strip()
                urls[idx] = f.readline().strip()
                n_steps[idx] = int(f.readline().strip())
                steps[idx] = f.readline().strip().split(',')
                next(f)
                idx = f.readline()
        return {'title': titles, 'url': urls, 'n_steps': n_steps, 'steps': steps}


    def get_vids(self, path):
        task_vids = {}
        with open(path,'r') as f:
            for line in f:
                task, vid, url = line.strip().split(',')
                task = int(task)
                if task not in task_vids:
                    task_vids[task] = []
                task_vids[task].append(vid)
        return task_vids
