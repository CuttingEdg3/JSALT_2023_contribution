import os
import json
import torch
import time
import re
import string
import random
import copy
import random
import shutil
import argparse
from difflib import SequenceMatcher

#os.chdir('/content/drive/MyDrive/PhD/UBAR/UBAR-MultiWOZ')

import os, random, argparse, time, logging, json, tqdm
import numpy as np



data_mada = json.loads(open('./data/multi-woz-processed/multi_act_mapping_train.json', 'r', encoding='utf-8').read().lower())

data = json.loads(open('./data/original/extra_data_for_damd.json', 'r', encoding='utf-8').read().lower())

data10 = []

with open('./10_percent_training.txt' , 'r') as f:
  lines = f.readlines()

  for line in lines:
    data10.append(line.strip())

print('Red data')



data_aug = {}
count = 0

for i in data:
  if i in data10 :
    count+=1
    print('Processing no '+str(count)+' '+i)
    logging.info('Processing no '+str(count)+' '+i)


    data_aug[i] = copy.deepcopy(data[i])
    for n, j in enumerate(data[i]['log']):
      choice = random.randint(0,3)
      print('choice for ',n, choice)

      if choice:


        #print(n,j['user'], j['sys_act'],j['resp'])


        try :
          resp_list = [r for _,r in list(data_mada[i][str(n)].items())]

          resp_list = [l for d in resp_list for l in d if l.split(':')[0].split('-')[0] in data10]

          resp_tup = []

          for ii in resp_list:

            resp_tup.append([ii.split(':')[0], ii.split(':')[1]])

          for it in resp_tup:
            #print(j['sys_act'],':', it[1],':',SequenceMatcher(None, j['sys_act'], it[1]).ratio())
            it.append(SequenceMatcher(None, j['sys_act'], it[1]).ratio())

          resp_tup = [(t1[0],t1[1],t1[2]) for t1 in resp_tup]


          resp_tup.sort(key=lambda tup: tup[2], reverse=True)

          #print(resp_tup)


          data_aug[i]['log'][n]['sys_act'] = data[resp_tup[0][0].split('-')[0]]['log'][int(resp_tup[0][0].split('-')[1])]['sys_act']
          data_aug[i]['log'][n]['resp']  = data[resp_tup[0][0].split('-')[0]]['log'][int(resp_tup[0][0].split('-')[1])]['resp']
          data_aug[i]['log'][n]['resp_pre_delex']  = data[resp_tup[0][0].split('-')[0]]['log'][int(resp_tup[0][0].split('-')[1])]['resp_pre_delex']
          # print(data[resp_tup[0][0].split('-')[0]]['log'][int(resp_tup[0][0].split('-')[1])]['sys_act'])
          # print(data_aug[i]['log'][n]['sys_act'])


        except Exception as e:

          print(n, 'No augmentation',e)

    #break
print('Done')
logging.info('Done')


with open("./data/augmentations/mada/data_for_damd.json", "w") as f:
  json.dump(data_aug, f, indent=2)
  logging.info('Done')
