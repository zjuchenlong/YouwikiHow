import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
# import torch
import re
from collections import defaultdict
import json
import pickle as pkl
import numpy as np

# Roberta For sentence feature extraction
# from transformers import RobertaTokenizer, RobertaModel

from article_simplification import get_headline_and_article, article_preprocessing

from dataloader.CrossTask import CrossTaskDataset
from dataloader.wikiHow_text import wikiHowTextDataset
from dataloader.HowTo100M import HowTo100MDataset

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


crosstask_dataset = CrossTaskDataset()
all_crosstask_tasks = crosstask_dataset.get_all_tasks()

howto_dataset = HowTo100MDataset()
howto_tasks   = howto_dataset.howto_tasks
howto_id2task = howto_dataset.howto_id2task 
howto_task2id = howto_dataset.howto_task2id
howto_task_id = howto_dataset.howto_task_id

wikiHow_text_dataset = wikiHowTextDataset()
wikihowAll = wikiHow_text_dataset.wikihowAll
wikihowSep = wikiHow_text_dataset.wikihowSep
wikiHow_tasks = wikiHow_text_dataset.wikiHow_tasks


print('Filter by exactly match:')
# howto_task_id_with_videos: 25,312 -> 25,086
##################################################################
exactly_match_tasks = defaultdict(list)
howto_task_id_with_videos = set(howto_task_id.to_list())
for howto_t_id in tqdm(howto_task_id_with_videos):
    if howto_t_id in howto_id2task.keys():
        howto_t = howto_id2task[howto_t_id]
        for wiki_t in wikiHow_tasks: # wikihow tasks: 215,364
            if howto_t in wiki_t:
                exactly_match_tasks[howto_t].append(wiki_t)
if not os.path.exists('cache'):
    os.makedirs('cache')
with open('cache/howto100m_match_tasks.json', 'w') as f:
    json.dump(exactly_match_tasks, f)
###################################################################
# exactly_match_tasks = json.load(open('cache/howto100m_match_tasks.json', 'r'))


print('Filter by video numbers:')
## 25,086 -> decrease to 2,535 tasks
video_stats = defaultdict(list)
rank_threshold = 50
number_threshold = 30
for task in tqdm(exactly_match_tasks.keys()):
    task_id = howto_task2id[task]
    task_videos = howto_dataset.get_videos_by_task_id(task_id)
    filtered_task_videos = task_videos[task_videos['rank'] < rank_threshold]
    if len(filtered_task_videos) >= number_threshold:
        video_stats[task_id] = filtered_task_videos['video_id'].to_list()


print('Filter by noisy text article:')
# remove some noisy headline & article
# 2,535 tasks -> 2,463 tasks (old 2,299 task)
##################################################################
all_simplified_text = dict()
for task_id in tqdm(video_stats.keys()):
    task_name = howto_id2task[task_id]
    all_simplified_text[task_id] = []
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
        all_simplified_text[task_id].append(task_full_name_dict)
all_simplified_text_temp1 = dict()
for task_k, task_v in all_simplified_text.items():
    if len(task_v) == 0:
        continue
    elif len(task_v) > 1:
        for v in task_v:
            if v['task_full_name'] in ['How to ' + howto_id2task[task_k], 'How to ' + howto_id2task[task_k] + '1']:
                all_simplified_text_temp1[task_k] = v
                break
    else:
        all_simplified_text_temp1[task_k] = task_v[0]
all_simplified_text = all_simplified_text_temp1
with open('cache/howto100m_all_simplified_text.pkl', 'wb') as f:
    pkl.dump(all_simplified_text, f)
##################################################################
# all_simplified_text = pkl.load(open('cache/howto100m_all_simplified_text.pkl', 'rb'))


# new 2,463 tasks -> 1,384 task
print('Filter by article length:')
headline_len_max_threshold = 10
headline_len_min_threshold = 1
article_len_max_threshold = 30
article_len_min_threshold = 1
filtered_simplified_text = dict()
for task_id, task_t in tqdm(all_simplified_text.items()):

    headline_len = len(task_t['task_text']['simplified_headline_list'])
    article_len = len(task_t['task_text']['simplified_article_list'])
    if (headline_len > headline_len_max_threshold) or \
        (headline_len < headline_len_min_threshold) or \
        (article_len > article_len_max_threshold) or \
        (article_len < article_len_min_threshold):
        continue
    else:
        filtered_simplified_text[task_id] = task_t

# add video to the filtered_simplified_text
processed_data = dict()
for task_id, task_text in tqdm(filtered_simplified_text.items()):

    assert task_id in video_stats
    processed_data[task_id] = dict()
    processed_data[task_id]['task_full_name'] = task_text['task_full_name']
    processed_data[task_id]['task_text'] = task_text['task_text']
    processed_data[task_id]['task_video'] = list()
    for vid in video_stats[task_id]:
        # filter out some videos
        if vid not in ['bHHB3u9pZj4', 'ZJNm0DaKLYs']:
            processed_data[task_id]['task_video'].append(vid)            


# get the duration of each video
wikihow_grounding_train_path = Path('/dvmm-filer3a/users/chenlong/Datasets/wikiHow_grounding/raw_videos/train')
all_video_duration = dict()
for task_id in tqdm(processed_data.keys()):
    videos = processed_data[task_id]['task_video']
    for vid in videos:

        mp4_vid_path = wikihow_grounding_train_path / (vid + '.mp4')
        webm_vid_path = wikihow_grounding_train_path / (vid + '.webm')

        if os.path.exists(mp4_vid_path):
            vid_path = mp4_vid_path
        elif os.path.exists(webm_vid_path):
            vid_path = webm_vid_path

        dur = _get_duration(vid_path)
        all_video_duration[vid] = dur

with open('cache/train_video_durations.json', 'w') as f:
    json.dump(all_video_duration, f)
# all_video_duration = json.load(open('cache/train_video_durations.json', 'r'))



for task_id, task_text in tqdm(filtered_simplified_text.items()):
    videos = processed_data[task_id]['task_video']
    processed_data[task_id]['video_duration'] = dict()
    for vid in videos:
        if vid not in all_video_duration:
            del processed_data[task_id]
            print(f'Task id {task_id} is removed!')
            break
        processed_data[task_id]['video_duration'][vid] = all_video_duration[vid]


# final 1,383 tasks
pkl.dump(processed_data, open('annotations/wikihow_data.pkl', 'wb'))

