#imports

import os
import json
import torch
import time
import re
import string
import random

#os.chdir('/content/drive/MyDrive/PhD/UBAR/UBAR-MultiWOZ')
from googletrans import Translator
import logging
import numpy as np
import argparse
from config import global_config as cfg
from new_train import Modal


#Functions
def clean_text(text):
    text = text.strip()
    text = text.lower()
    text = text.replace(u"’", "'")
    text = text.replace(u"‘", "'")
    text = text.replace(';', ',')
    text = text.replace('"', ' ')
    text = text.replace('/', ' and ')
    text = text.replace("don't", "do n't")
    text = clean_time(text)
    baddata = { r'c\.b (\d), (\d) ([a-z])\.([a-z])': r'cb\1\2\3\4',
                        'c.b. 1 7 d.y': 'cb17dy',
                        'c.b.1 7 d.y': 'cb17dy',
                        'c.b 25, 9 a.q': 'cb259aq',
                        'isc.b 25, 9 a.q': 'is cb259aq',
                        'c.b2, 1 u.f': 'cb21uf',
                        'c.b 1,2 q.a':'cb12qa',
                        '0-122-336-5664': '01223365664',
                        'postcodecb21rs': 'postcode cb21rs',
                        r'i\.d': 'id',
                        ' i d ': 'id',
                        'Telephone:01223358966': 'Telephone: 01223358966',
                        'depature': 'departure',
                        'depearting': 'departing',
                        '-type': ' type',
                        r"b[\s]?&[\s]?b": "bed and breakfast",
                        "b and b": "bed and breakfast",
                        r"guesthouse[s]?": "guest house",
                        r"swimmingpool[s]?": "swimming pool",
                        "wo n\'t": "will not",
                        " \'d ": " would ",
                        " \'m ": " am ",
                        " \'re' ": " are ",
                        " \'ll' ": " will ",
                        " \'ve ": " have ",
                        r'^\'': '',
                        r'\'$': '',
                                }
    for tmpl, good in baddata.items():
        text = re.sub(tmpl, good, text)

    text = re.sub(r'([a-zT]+)\.([a-z])', r'\1 . \2', text)   # 'abc.xyz' -> 'abc . xyz'
    text = re.sub('([.,!?()])', r' \1 ', text)   # if 'abc. ' -> 'abc . '
    #text = re.sub(r'(\w+)\.\.', r'\1 . ', text)    # if 'abc.' -> 'abc . '
    text = re.sub(' +', ' ',text)


    with open('data/multi-woz/mapping.pair', 'r') as fin:
        for line in fin.readlines():
            fromx, tox = line.replace('\n', '').split('\t')
            text = ' ' + text + ' '
            text = text.replace(' ' + fromx + ' ', ' ' + tox + ' ')[1:-1]

    return text


def clean_time(utter):
    utter = re.sub(r'(\d+) ([ap]\.?m)', lambda x: x.group(1) + x.group(2), utter)   # 9 am -> 9am
    utter = re.sub(r'((?<!\d)\d:\d+)(am)?', r'0\1', utter)
    utter = re.sub(r'((?<!\d)\d)am', r'0\1:00', utter)
    utter = re.sub(r'((?<!\d)\d)pm', lambda x: str(int(x.group(1))+12)+':00', utter)
    utter = re.sub(r'(\d+)(:\d+)pm', lambda x: str(int(x.group(1))+12)+x.group(2), utter)
    utter = re.sub(r'(\d+)a\.?m',r'\1', utter)
    return utter



def translate_back(input_pre_delex, input_init, delex_dict, isUser, turn_num, dial_id):
  translator = Translator()
  input = input_init
  splt = input.split()
  #delex_dict = {v: k for k, v in delex_dict.items()}

  delex_tokens = {}
  for n, token in enumerate(splt):
    if token.startswith('[') and token.endswith(']'):
      delex_tokens['[{}]'.format(n)] = splt[n]
      splt[n] = '[{}]'.format(n)

  input = ' '.join(splt)

  results = translator.translate(input,dest='fr').text
  #print(results)
  #results = translator.translate(results,dest='es').text

  #results = translator.translate(results,dest='de').text

  results = translator.translate(results,dest='en').text

  #print(results)


  for delex in delex_tokens:
    #print(delex , delex_tokens[delex])
    results = results.replace(delex, delex_tokens[delex])
  

  original_augmented = results
  for i in delex_dict:
    if not isUser and delex_dict[i] not in results:
      logging.info('With system speaking did not include all information')
      return input_pre_delex, input_init      

    original_augmented = original_augmented.replace(delex_dict[i],i)
  original_augmented = clean_text(original_augmented.lower())
  #print('results',original_augmented)
  if isUser and not m.check_bspan(dial = dial_id, turn_num = turn_num, augmentation = original_augmented):
    print(original_augmented)
    logging.info('With user speaking produces different bspan/aspan')
    
    return input_pre_delex, input_init

  return clean_text(original_augmented.lower()).strip(), clean_text(results.lower()).strip()




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

def parse_arg_cfg(args):
    # add args to cfg
    if args.cfg:
        for pair in args.cfg:
            k, v = tuple(pair.split('='))
            dtype = type(getattr(cfg, k))
            if dtype == type(None):
                raise ValueError()
            if dtype is bool:
                v = False if v == 'False' else True
            elif dtype is list:
                v = v.split(',')
                if k == 'cuda_device':
                    v = [int(no) for no in v]
            else:
                v = dtype(v)
            setattr(cfg, k, v)
    return


def get_model() :
  args = argparse.Namespace(mode='test', cfg=['eval_load_path=/fsx/workspace/cvlachos/projects/UBAR/UBAR-MultiWOZ/experiments/10_percent_data/epoch59_trloss1.0552_gpt2', 'use_true_prev_bspn=False', 'use_true_prev_aspn=False', 'use_true_db_pointer=False', 'use_true_prev_resp=False', 'use_true_curr_bspn=False', 'use_true_curr_aspn=False', 'use_all_previous_context=True'])

  cfg.mode = 'test'

  parse_arg_cfg(args)
  # cfg.model_path = cfg.eval_load_path
  cfg.gpt_path = cfg.eval_load_path

  cfg._init_logging_handler('test')
  if cfg.cuda:
      if len(cfg.cuda_device) == 1:
          cfg.multi_gpu = False
          # torch.cuda.set_device(cfg.cuda_device[0])
          device = torch.device("cuda:{}".format(cfg.cuda_device[0]))
      else:
          pass  # multi-gpu
  else:
      device = torch.device('cpu')
      logging.info('Device: {}'.format(torch.cuda.current_device()))

  # fix random seed
  torch.manual_seed(cfg.seed)
  torch.cuda.manual_seed(cfg.seed)
  random.seed(cfg.seed)
  np.random.seed(cfg.seed)

  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  print('new device',device)
  m = Modal(device)

  return m


#Run Augmentation
m = get_model()

logging.info('Reading data')
data10 = []

with open('./10_percent_training.txt' , 'r')as f:
  lines = f.readlines()

  for line in lines:
    data10.append(line.strip())

k = 0
count = 0
data_json = json.dumps({})


with open("./data/augmentations/data_for_damd.json", "w") as outfile:
    outfile.write(data_json)

data = json.loads(open('./data/original/extra_data_for_damd.json', 'r', encoding='utf-8').read().lower())


logging.info('Running augmentation')
while k < len(data) :

  try :
    for n, i in enumerate(list(data.keys())[k:]):
      #print(k)
      #print(n,i)

      if i in data10 :
        logging.info('Dialog '+i)
        for turn_num, turn in enumerate(data[i]['log']):
          logging.info('Turn '+str(turn_num))

          print(turn['user'], turn['user_delex'])          
          turn['user'], turn['user_delex'] = translate_back(turn['user'], turn['user_delex'], get_delex_dict(turn['user'].split(), turn['user_delex'].split()), True, turn_num, i)
          print(turn['user'], turn['user_delex'])
          
          print()
          print(turn['resp_pre_delex'], turn['resp'])
          turn['resp_pre_delex'], turn['resp'] = translate_back(turn['resp_pre_delex'], turn['resp'], get_delex_dict(turn['resp_pre_delex'].split(), turn['resp'].split()), False, turn_num,i )
          print(turn['resp_pre_delex'], turn['resp'])
        
        #asdfbshjdfk()


        logging.info('Saving')

        data_aug = json.loads(open('./data/augmentations/data_for_damd.json', 'r', encoding='utf-8').read().lower())
        data_aug[i] = data[i]


        with open("./data/augmentations/data_for_damd.json", "w") as f:
          json.dump(data_aug, f, indent=2)
          count += 1
          logging.info('Have saved'+str( count))

      k+=1

  except Exception as e :
    logging.info('Something bad happened. Saving and waiting.',e)
  
  finally :
    break
    

logging.info('Done')
