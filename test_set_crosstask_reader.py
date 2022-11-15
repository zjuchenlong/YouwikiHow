from pathlib import Path
import pickle as pkl
from tqdm import tqdm
import pickle as pkl
import json
import os

from dataloader.CrossTask import CrossTaskDataset
from dataloader.HowTo100M import HowTo100MDataset
from article_simplification import get_headline_and_article, article_preprocessing

import ffmpeg
import subprocess

def _get_duration(video_path):
    """
    https://stackoverflow.com/questions/31024968/using-ffmpeg-to-obtain-video-durations-in-python
    Get the duration of a video using ffprobe.
    """
    cmd = 'ffprobe -i {} -show_entries format=duration -v quiet -of csv="p=0"'.format(video_path)
    output = subprocess.check_output(
        cmd,
        shell=True, # Let this run in the shell
        stderr=subprocess.STDOUT
    )
    # return round(float(output))  # ugly, but rounds your seconds up or down
    return float(output)


howto_dataset = HowTo100MDataset()
howto_tasks   = howto_dataset.howto_tasks
howto_id2task = howto_dataset.howto_id2task
howto_task2id = howto_dataset.howto_task2id
howto_task_id = howto_dataset.howto_task_id

crosstask_dataset = CrossTaskDataset()
all_crosstask_tasks = crosstask_dataset.get_primary_tasks()

## howto_task_id_with_videos: 25,312 -> 25,086
# howto_task_id_with_videos = set(howto_task_id.to_list())
# for howto_t_id in tqdm(howto_task_id_with_videos):
#     if howto_t_id in howto_id2task.keys():
#         howto_t = howto_id2task[howto_t_id]
#         for wiki_t in wikiHow_tasks: # wikihow tasks: 215,365
#             if howto_t in wiki_t:
#                 exactly_match_tasks[howto_t].append(wiki_t)
exactly_match_tasks = json.load(open('cache/howto100m_match_tasks.json', 'r'))


# borrow from task_set_from_crosstask.py
all_crosstask_text = dict()
for task_name in tqdm(all_crosstask_tasks):

    all_crosstask_text[task_name] = []
    for task_full_name in exactly_match_tasks[task_name]:
        success, headline_list, article_list = get_headline_and_article(task_full_name)
        if not success:
            break
        simplified_headline_list, simplified_article_list, head2sent, sent2head \
                            = article_preprocessing(headline_list, article_list)
        task_full_name_dict = {}
        task_full_name_dict['task_full_name'] = task_full_name
        task_full_name_dict['task_text'] = {}
        task_full_name_dict['task_text']['simplified_headline_list'] = simplified_headline_list
        task_full_name_dict['task_text']['simplified_article_list'] = simplified_article_list
        task_full_name_dict['task_text']['head2sent'] = head2sent
        task_full_name_dict['task_text']['sent2head'] = sent2head
        all_crosstask_text[task_name].append(task_full_name_dict)
all_crosstask_text_temp1 = dict()
for task_k, task_v in all_crosstask_text.items():
    if len(task_v) == 0:
        continue
    elif len(task_v) > 1:
        for v in task_v:
            if v['task_full_name'] in ['How to ' + task_k, 'How to ' + task_k + '1']:
                all_crosstask_text_temp1[task_k] = v
                break
    else:
        all_crosstask_text_temp1[task_k] = task_v[0]
all_crosstask_text = all_crosstask_text_temp1


def assert_list(list_a, list_b):

    assert len(list_a) == len(list_b)
    for l_a, l_b in zip(list_a, list_b):
        assert l_a == l_b


def read_step_annotaions_from_CrossTask(task_id):

    step_annotation = dict()
    crosstask_videos = crosstask_dataset.videos[task_id]
    for vid in crosstask_videos:
        key = str(task_id) + '_' + vid
        assert key in crosstask_dataset.all_step_annotations.keys()
        annos = crosstask_dataset.all_step_annotations[key]
        step_annotation[vid] = annos

    return step_annotation



crosstask_test_set_path = Path('./cache/manually_test_set_from_crosstask.txt')

crosstask_test_set = dict()
with open(crosstask_test_set_path,'r') as f:
    task_name = f.readline().strip()
    while task_name != '':

        task_id = howto_task2id[task_name]
        crosstask_test_set[task_id] = dict()

        step_num = int(f.readline().strip())
        step_list = list()
        for i in range(step_num):
            step = f.readline().strip()
            step = ','.join(step.split(',')[1:])
            step_list.append(step)
        task_full_name = f.readline().strip()
        assert task_full_name == all_crosstask_text[task_name]['task_full_name']

        headline_num = int(f.readline().strip())
        headline_list = list()
        for i in range(headline_num):
            headline = f.readline().strip()
            headline = ','.join(headline.split(',')[1:])
            headline_list.append(headline)
        assert_list(headline_list, all_crosstask_text[task_name]['task_text']['simplified_headline_list'])

        article_num = int(f.readline().strip())
        article_list = list()
        for i in range(article_num):
            article = f.readline().strip()
            article = ','.join(article.split(',')[1:])
            article_list.append(article)
        assert_list(article_list, all_crosstask_text[task_name]['task_text']['simplified_article_list'])

        step2headline = f.readline().strip()
        each_step_gt = step2headline.split(',')[:-1]


        # same format as the train set
        crosstask_test_set[task_id]['task_full_name'] = task_full_name
        crosstask_test_set[task_id]['task_text'] = all_crosstask_text[task_name]['task_text']
        crosstask_test_set[task_id]['task_video'] = crosstask_dataset.videos[task_id]

        # extra annotations in the test set
        crosstask_test_set[task_id]['step2headline'] = each_step_gt
        crosstask_test_set[task_id]['crosstask_annotations'] = read_step_annotaions_from_CrossTask(task_id)

        # dummy steps
        next(f) # for step2article
        next(f)
        next(f)        

        task_name = f.readline().strip()


# get the duration of each video
#########################################################################
wikihow_grounding_test_path = Path('/dvmm-filer3a/users/chenlong/Datasets/wikiHow_grounding/raw_videos/test')
all_video_duration = dict()
for task_id in tqdm(crosstask_test_set.keys()):
    videos = crosstask_test_set[task_id]['task_video']
    for vid in videos:

        mp4_vid_path = wikihow_grounding_test_path / (vid + '.mp4')
        webm_vid_path = wikihow_grounding_test_path / (vid + '.webm')

        if os.path.exists(mp4_vid_path):
            vid_path = mp4_vid_path
        elif os.path.exists(webm_vid_path):
            vid_path = webm_vid_path
        else:
            raise ValueError

        dur = _get_duration(vid_path)
        all_video_duration[vid] = dur

with open('cache/test_video_durations.json', 'w') as f:
    json.dump(all_video_duration, f)
#########################################################################
# all_video_duration = json.load(open('cache/test_video_durations.json', 'r'))


for task_id, task_text in tqdm(crosstask_test_set.items()):
    videos = crosstask_test_set[task_id]['task_video']
    crosstask_test_set[task_id]['video_duration'] = dict()
    for vid in videos:
        crosstask_test_set[task_id]['video_duration'][vid] = all_video_duration[vid]


pkl.dump(crosstask_test_set, open('annotations/crosstask_test.pkl', 'wb'))

