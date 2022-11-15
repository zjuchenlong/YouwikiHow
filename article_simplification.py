# import torch 
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import numpy as np
from collections import defaultdict

import re
import spacy
from nltk import tokenize

from dataloader.wikiHow_text import wikiHowTextDataset

## SRL labeling
# from allennlp.predictors.predictor import Predictor
# import allennlp_models.tagging
# SRL_predictor = Predictor.from_path("/dvmm-filer3a/users/chenlong/Models/AllenNLP/structured-prediction-srl-bert.2020.12.15.tar.gz")

## spaCy parser
spacy_parser = spacy.load("en_core_web_sm")


wikiHow_text_dataset = wikiHowTextDataset()
wikihowAll = wikiHow_text_dataset.wikihowAll
wikihowSep = wikiHow_text_dataset.wikihowSep
wikiHow_tasks = wikiHow_text_dataset.wikiHow_tasks


def find_verb(sentence):

    spacy_parse_output = spacy_parser(sentence)

    verb_list = []
    for token in spacy_parse_output:
        if token.pos_ == 'VERB':
            verb_list.append(token.text)

    return verb_list


def append_list_with_check(item_list, item):
    if item not in item_list:
        item_list.append(item)
    return item_list


def sentence_simplification(sentence):
    
    # remove sentence between [] and ()
    sentence = re.sub("([\(\[]).*?([\)\]])", "", sentence)
    sentence = ' '.join([x for x in sentence.split(' ') if x != '']) # remove space

    spacy_parse_output = spacy_parser(sentence)

    mask_list = list()
    for entity in spacy_parse_output.noun_chunks:

        # filter out words not in the children, eg: 'cut out two [identically-]sized wooden panels for each shelf.'
        entity_span_start, entity_span_end = entity.start, entity.end
        all_child_root_idx = [e.i for e in entity.root.children]
        all_child_root_idx.append(entity.root.i)
        for i in range(entity_span_start, entity_span_end):
            if i not in all_child_root_idx:
                append_list_with_check(mask_list, i)

        # 'Measure out [3 tablespoon of] milk and mix it into the bowl of eggs.'
        for x in entity.root.children:

            if x.pos_ == 'NUM':

                x_idx = sorted(all_child_root_idx).index(x.i)

                # "Cut out at least [two] long[] 1-by-2-inch strips of[] wood."
                if x_idx < (len(all_child_root_idx) - 1) and spacy_parse_output[sorted(all_child_root_idx)[x_idx + 1]].pos_ == 'ADJ':
                    append_list_with_check(mask_list, x.i)

                else:
                    for y in entity.root.children:
                        append_list_with_check(mask_list, y.i)

        # remove the whole entity
        # 'add [a cup of] surger or [a dash of] honey'
        entity_child = [x for x in entity.root.children]
        if len(entity_child) >= 2:
            remove_state = True
                # 'Measure out [3] tablespoon[root] [of] milk and mix it into the bowl of eggs.' # len(entity_child) == 2
                # 'After [a] minute[root] or [two] [as] the machine pre-heats the water'          # len(entity_child) == 3
                # **** Neg example ***
                # 'Cut out at least [two] long[] 1-by-2-inch strips[root] of[] wood.'
            for i in range(len(entity_child) - 1):
                if not ((entity_child[i].dep_ in ['compound', 'nummod'] or entity_child[i].text in ['a', 'an']) & 
                    (entity_child[-1].dep_ == 'prep')
                    ):
                    remove_state = False
                    break
            if remove_state:
                append_list_with_check(mask_list, entity_child[0].i)
                append_list_with_check(mask_list, entity_child[1].i)
                append_list_with_check(mask_list, entity.root.i)


    for token in spacy_parse_output:

        # 'though this is not recommended as the beans lose about 60% of aroma [after 15 minutes] and you lose a significant amount of flavor.'
        if token.pos_ == 'ADP':
            for x in token.children:
                if x.pos_ == 'NUM':
                    for y in token.children:
                        append_list_with_check(mask_list, y.i)
                    append_list_with_check(mask_list, token.i)
                    break

    # remove extra words
    for token in spacy_parse_output:

        # remove extra "concatation words", if the word before/after is deleted
        if token.dep_ == 'cc':
            # 'After a minute [or] two as the machine pre-heats the water,
            if (token.i - 1) in mask_list and (token.i + 1) in mask_list:
                append_list_with_check(mask_list, token.i)

        if str(token) == '-':
            if (token.i - 1) in mask_list and token.i in mask_list:
                append_list_with_check(mask_list, token.i + 1)
            if (token.i + 1) in mask_list and token.i in mask_list:
                append_list_with_check(mask_list, token.i - 1)

        # remove the root if all subtree nodes have been cleaned        
        # 'though this is not recommended as the beans lose about 60% of aroma [after] 15 minutes and you lose a significant amount of flavor.'
        if token.pos_ == 'ADP': # only consider ADP type
            entity_subtree_child = [x for x in token.subtree]
            ROOT_DELETE = False
            if len(entity_subtree_child) > 1:
                ROOT_DELETE = True
                for x in entity_subtree_child:
                    if (x.i != token.i) and (x.i not in mask_list):
                        ROOT_DELETE = False
                        break

            if ROOT_DELETE:
                append_list_with_check(mask_list, token.i)

    # finish process in dependency parsing
    filtered_sentence = list()
    for token in spacy_parse_output:
        if token.i not in mask_list:
            filtered_sentence.append(token.text)
    filtered_sentence = " ".join(filtered_sentence)
    filtered_sentence = re.sub(r'\s([.,!])', '\g<1>', filtered_sentence) # remove space between .,! etc.
    filtered_sentence = re.sub(r'\s([-/])\s', '\g<1>', filtered_sentence) # remove space between -/ .
    filtered_sentence = re.sub(r' \'', '\'', filtered_sentence) # remove space before \', such as ``you 're".

    return filtered_sentence

    # start SRL
    # SRL_prediction = SRL_predictor.predict(sentence=filtered_sentence)
    
    # all_sentence_words = SRL_prediction['words']
    # words_mask = [1] * len(all_sentence_words)

    # spacy_verbs = find_verb(filtered_sentence)

    # for verb_frame in SRL_prediction['verbs']:
        
    #     if verb_frame['verb'] in spacy_verbs:

    #         # Keep -V, -ARG0, -ARG1, O
    #         for i, tag in enumerate(verb_frame['tags']):
    #             if ('-V' in tag) or ('-ARG0' in tag) or ('-ARG1' in tag) or ('O' in tag):
    #                 pass
    #             else:
    #                 words_mask[i] = 0
        
    # return " ".join(list(np.array(all_sentence_words)[np.array(words_mask) == 1]))



def get_headline_and_article(task_name):

    wkhAll_frame = wikihowAll.loc[wikihowAll['title'] == task_name]
    all_headline, all_article = wkhAll_frame['headline'].values[0], wkhAll_frame['text'].values[0]

    # Some extreme cases have extra "\n,\n"
    all_headline = re.sub(r'\.,', '.\n', all_headline)
    all_headline = re.sub(r'\n,', '\n', all_headline)
    all_headline = re.sub(r',\n', '\n', all_headline)
    all_headline = re.sub(r'\n+', '\n', all_headline)

    wkhSep_frames = wikihowSep.loc[wikihowSep['title'] == task_name]

    headline_list = all_headline.split('\n')[1:]
    headline_list = list(filter(lambda headline: len(headline) > 2, headline_list))


    if len(headline_list) == len(wkhSep_frames):
        article_list = list()
        for idx, hl in enumerate(headline_list):
            frame = wkhSep_frames.iloc[idx]

            if hl not in frame['headline']:
                return False, None, None
            assert hl in frame['headline']
            article = frame['text']
            article_list.append(article)

        return True, headline_list, article_list
    else:
        return False, None, None

def article_preprocessing(headline_list, article_list):

    """
    Some article in article_list is "nan"
    """
    assert len(headline_list) == len(article_list) # raw format from wikiHow dataset
    # simplified articiles
    simplified_headline_list = list()
    simplified_article_list = list()

    for hl in headline_list:
        simplified_hl = sentence_simplification(hl)
        simplified_headline_list.append(simplified_hl)

    head2sent = defaultdict(list)
    sent2head = defaultdict(int)

    for head_idx, art in enumerate(article_list):

        if art == 'nan':
            head2sent[head_idx] = -1
            continue

        # remove text after "\n\n\n\n\n XXXX "
        start_idx = [i.start() for i in re.finditer("\n\n\n", art)]
        if len(start_idx) > 0:
            art = art[0:start_idx[0]]

        # remove text between "\n\n XXXX \n\n"
        art = re.sub("\n\n.*?\n\n", "", art)

        # replace two punctuations into single
        art = re.sub("\.;", ".", art)

        # split article into sentences
        sent_list = tokenize.sent_tokenize(art)

        for sent in sent_list:
            sent = re.sub("\n", " ", sent)
            simplified_sent = sentence_simplification(sent)

            # filter out some simplified_sent
            if '.jpg' in simplified_sent:
                continue
            if len(simplified_sent) < len(simplified_headline_list[head_idx]): # remove over-short sentence (shorter than headline)
                continue
            # if simplified_sent.startwith('If'):
            #     import pdb; pdb.set_trace()
            #     continue
            else:
                simplified_article_list.append(simplified_sent)
                sent_idx = len(simplified_article_list) - 1
                sent2head[sent_idx] = head_idx
                head2sent[head_idx].append(sent_idx)

    return simplified_headline_list, simplified_article_list, \
            head2sent, sent2head



if __name__=="__main__":

    demo_sentences = [
    "Measure out 3 tablespoon of milk and mix it into the bowl of eggs.",
    "Then, mix in 1 teaspoon (4.9 mL) of vanilla extract, followed by 1 teaspoon (2.6 grams) of cinnamon.",
    "add a cup of surger or a dash of honey.",
    "If you want a sweeter taste, you can add ¼ cup (32 grams) of sugar or a dash of honey to the mixture as well.",
    "Measure out 2⁄3 tablespoon (9.9 mL) of milk and mix it into the bowl of eggs. Then, mix in 1 teaspoon (4.9 mL) of vanilla extract, followed by 1 teaspoon (2.6 grams) of cinnamon.",
    "Drop around 1 tablespoon (15 mL) of butter on the middle of the pan so that it can spread out evenly as it melts.",
    "you can also use pre-ground coffee, though this is not recommended as the beans lose about 60% of aroma after 15 minutes and you lose a significant amount of flavor.",
    "most coffee makers like to have about 2 tablespoons (30\xa0ml) per cup",
    "After a minute or two as the machine pre-heats the water, your coffee should begin brewing.",
    "though this is not recommended as the beans lose about 60% of aroma after 15 minutes and you lose a significant amount of flavor.",
    "2. Add milk, vanilla extract, and cinnamon to the bowl. Measure out 2⁄3 tablespoon (9.9 mL) of milk and mix it into the bowl of eggs. Then, mix in 1 teaspoon (4.9 mL) of vanilla extract, followed by 1 teaspoon (2.6 grams) of cinnamon. It's important to add the ingredients separately, since delicate eggs whites need to incorporate ingredients one at a time.",
    "Cut out two identically-sized wooden panels for each shelf.",
    "Attach all three 1-by-2-inch strips to one of the two panels.",
    "This is important because the temperature at which alcohol evaporates is 78.6 degrees C",
    "To make approximately 32 1-ounce shots, all you'll need is:",
    "Cut out at least two long 1-by-2-inch strips of wood."
    ]

    for sentence in demo_sentences:
        simplified_sentence = sentence_simplification(sentence)
        print("***** Original Sentence *****:", sentence)
        print("***** Simplified Sentence *****:", simplified_sentence)
        print("\n")



    # task_name = 'How to Make Coffee1'
    # Success, headline_list, article_list = get_headline_and_article(task_name)
    # simplified_headline_list, simplified_article_list, head2sent, sent2head \
    #         = article_preprocessing(headline_list, article_list)