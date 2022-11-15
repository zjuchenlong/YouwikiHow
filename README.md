# YouwikiHow

The repo contains the **YouwikiHow** dataset and original scripts to build this dataset.

## Download the Processed Datasets

### Dataset layout
```
YouwikiHow/
|   +-- annotations/
|       +-- wikihow_data.pkl
|       +-- crosstask_test.pkl
|   +-- features/
|       +-- train.csv
|       +-- test.csv
|       +-- train_s3d_features_from_ht100m
|       +-- test_s3d_features_fr16_sz256_nf16
```

### Training Set
The training set `wikihow_data.pkl` is a dictionary, **key** is the **wikiHow ID**, and `value` are:

- **task_full_name**: wikiHow task full name.
- **task_text**: wikihow articles. 
    - **simplified_headline_list**: List of all high-level summary sentences.
    - **simplified_article_list**:  list of all low-level articles.
    - **head2sent**: The sentence IDs mapping from *high-level* (key) sentences to *low-level* (value) sentences.
    - **sent2head**: The sentence IDs mapping from *low-level* (key) sentences to *high-level* (value) sentences.
- **task_video**: The list of all YouTube videos correspoinding to this wikiHow task.
- **video_duration**: The total duration of all YouTube videos.

### Test Set
The test set `crosstask_test.pkl` is same format as the training set, with two more keys:
- **step2headline**: The mannually mapping between the CrossTask steps (in original datasets) and the wikiHow articles (i.e., headlines).
- **crosstask_annotations**: The ground-truth for evaluation. *Keys* are the video ids, and *Values* are all possible ground-truth annotations propagated from CrossTask.

### Download Visual Features
- **train_s3d_features_from_ht100m** (coming soon)
- **test_s3d_features_fr16_sz256_nf16** (coming soon)

## Build from Scratch (Original Datasets)

### Download Original Datasets

- **[wikiHow](https://github.com/mahnazkoupaee/WikiHow-Dataset)**: Replace the path in `dataloader/wikiHow_text.py` with corresponding wikiHow dataset path.
- **[CrossTask](https://github.com/DmZhukov/CrossTask)**: Replace the path in `dataloader/CrossTask.py` with corresponding CrossTask dataset path.
- **[HowTo100M](https://www.di.ens.fr/willow/research/howto100m/)**: Replace the path in `datasetloader/HowTo100M.py` with corrsponding HowTo100M dataset path.

### YouwikiHow Preprocessing
1. Use manually rules to filter HowTo100M tasks for the training set.
```
python task_selection.py
```
2. Save the manually mapped crosstask annotations as the training set
```
python test_set_crosstask_reader.py
```
3. Generate S3D features using [S3D_Feature_Extractors](https://github.com/zjuchenlong/S3D_Feature_Extractors) with csv files in `features/train.csv` and `features/test.csv`.



## Citations
```
@inproceedings{chen2022weakly,
  title={Weakly-Supervised Temporal Article Grounding},
  author={Chen, Long and Niu, Yulei and Chen, Brian and Lin, Xudong and Han, Guangxing and Thomas, Christopher and Ayyubi, Hammad and Ji, Heng and Chang, Shih-Fu},
  booktitle={Empirical Methods in Natural Language Processing (EMNLP), 2022},
  year={2022}
}
```