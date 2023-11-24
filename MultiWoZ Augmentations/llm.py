import os
import json
import torch
import time
import re
import string
import random
import copy
import random
import logging



def llm_aug(input_pre_delex, input_init, delex_dict, isUser, turn_num, dial_id):
  #print(delex_dict)

  aug_lex = random.choice(dial_aug[dial_id][str(turn_num)][isUser])
  aug_lex = re.sub(r'([0-9]+) \. ([0-9])', r'\1.\2', aug_lex)
  aug_lex = re.sub(r'([0-9]+) \, ([0-9])', r'\1,\2', aug_lex)


  aug_delex = copy.deepcopy(aug_lex)

  for i in delex_dict :
    aug_delex = aug_delex.replace(i,delex_dict[i])


  return aug_lex, aug_delex






def get_delex_dict(text_splt, delex_splt):
  delex_tokens = {}

  if text_splt[-1] not in string.punctuation :
    text_splt.append('.')

  if delex_splt[-1] not in string.punctuation :
    delex_splt.append('.')

  n = 0
  delex_n = 0
  while n < len(text_splt):

    dict_str = ''
    #print(text_splt[n], '==', delex_splt[delex_n])

    if delex_n >= len(delex_splt):
      return delex_tokens


    if text_splt[n] == delex_splt[delex_n]:
      n += 1
      delex_n += 1
      dict_str = ''

    else :
      delex_n += 1
      dict_str += text_splt[n]+' '

      for k in range(n+1, len(text_splt)) :
        if text_splt[k] != delex_splt[delex_n]:
          dict_str += text_splt[k]+' '

        else :
          delex_tokens[dict_str.strip()] = delex_splt[delex_n-1]
          n = k
          break

  return delex_tokens



  logging.info('Reading data')
data10 = []

with open('./10_percent_training.txt' , 'r')as f:
  lines = f.readlines()

  for line in lines:
    data10.append(line.strip())

k = 0
count = 0
dial_aug = json.loads(open('./data/augmentations/llm/chatgpt_para.json', 'r', encoding='utf-8').read().lower())


data = json.loads(open('./data/original/extra_data_for_damd.json', 'r', encoding='utf-8').read().lower())


logging.info('Running augmentation')





#data = json.loads(open('./data/original/extra_data_for_damd.json', 'r', encoding='utf-8').read().lower())

data_aug = {}


for n, i in enumerate(list(data.keys())[:]):
  if i in data10 :
    print(i)
    for turn_num, turn in enumerate(data[i]['log']):
      print(turn_num)

      print(turn['user'], turn['user_delex'])
      turn['user'], turn['user_delex'] = llm_aug(turn['user'], turn['user_delex'], get_delex_dict(turn['user'].split(), turn['user_delex'].split()), 'user', turn_num, i)
      print(turn['user'],'||', turn['user_delex'])


      print()
      print(turn['resp_pre_delex'], turn['resp'])
      turn['resp_pre_delex'], turn['resp'] = llm_aug(turn['resp_pre_delex'], turn['resp'], get_delex_dict(turn['resp_pre_delex'].split(), turn['resp'].split()), 'resp_pre_delex', turn_num,i )
      print(turn['resp_pre_delex'],'||', turn['resp'])

      print()
    #asdfbshjdfk()

    data_aug[i] = data[i]





with open("./data/augmentations/llm/data_for_damd.json", "w") as f:
  json.dump(data_aug, f, indent=2)






print('Done')
