#imports
from transformers import RobertaTokenizerFast, RobertaForMaskedLM
import torch

tokenizer = RobertaTokenizerFast.from_pretrained("roberta-large")
model = RobertaForMaskedLM.from_pretrained("roberta-large")

import copy
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
punc = string.punctuation

stop_words = set(stopwords.words('english'))

import os
import json
import torch
import time
import re
import string
import random

#os.chdir('/content/drive/MyDrive/PhD/UBAR/UBAR-MultiWOZ')

import logging
import numpy as np
import argparse
from config import global_config as cfg
from new_train import Modal


torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'


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

domains = ['hotel','restaurant','hospital', 'taxi', 'police','bus', 'attraction', 'train']

def roberta_paraphrase(input_text, input_text_delex, const, isUser, turn_num, dial_id, prev, next):
  print()
  print(prev)
  print(input_text)
  print(next)

  while len(tokenizer.encode(prev +input_text+ next)) > 500:
    print('Input too long', len(tokenizer.encode(prev +input_text+ next)))
    if len(next) > len(prev):
      next = ' '.join(next.split()[10:])
    else :
      prev = ' '.join(prev.split()[10:])


  input_text = input_text.replace('-',' ')
  input_text_delex = input_text_delex.replace('-',' ')
  const = const.split()
  #print()
  # print(input_text)
  # print(input_text_delex)


  to_augment = []
  augmented = input_text
  augmented_delex = input_text_delex

  try :
  #if True :
    for i in input_text_delex.split() :
      if i not in punc and i not in domains and i not in stop_words and not i.startswith('[') and i not in const and not i.isnumeric() and i not in ['am','pm']:
        to_augment.append(i)
    #print(to_augment)

    for i in to_augment:
      #print(i)
      inputs = tokenizer(prev+input_text.replace(i,'<mask>')+next, return_tensors="pt")
      #print(input_text)

      with torch.no_grad():
          logits = model(**inputs).logits

      mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
      t=logits[0, mask_token_index].softmax(axis=-1).sort(descending = True)
      scores = t[0][0][:5]
      tokens = t[1][0][:5]

      output = []

      for n, ii in enumerate(tokens):
        output.append({'token_str' : tokenizer.decode(ii), 'score' : scores[n].item() })


      #print(output)

      for j in output:
        #print(j['token_str'], j['score'])

        if j['token_str'].lower().strip() != i and j['score'] > 0.00 and j['token_str'].lower().strip() not in punc and not j['token_str'].lower().strip().isnumeric():
          #print(j['token_str'].lower(), i)
          print('replacing',i,'with',j['token_str'].lower().strip())
          #print(m.check_bspan(dial = dial_id, turn_num = turn_num, augmentation = augmented.replace(i, j['token_str'].lower().strip())))
          if not isUser or (isUser and  m.check_bspan(dial = dial_id, turn_num = turn_num, augmentation = augmented.replace(i, j['token_str'].lower().strip()))):

            augmented = augmented.replace(i, j['token_str'].lower().strip())
            augmented_delex = augmented_delex.replace(i, j['token_str'].lower().strip())
            break

  except Exception as e :
    print('Error found, no augmentation will take place.!!!!!!!!!!!!!!!!!!',e)
  finally :
    return clean_text(augmented).lower(), clean_text(augmented_delex).lower()


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
  args = argparse.Namespace(mode='test', cfg=['eval_load_path=/fsx/workspace/cvlachos/projects/UBAR/UBAR-MultiWOZ/experiments/UBAR_augmentation_10/10_percent_data/epoch59_trloss1.0552_gpt2', 'use_true_prev_bspn=False', 'use_true_prev_aspn=False', 'use_true_db_pointer=False', 'use_true_prev_resp=False', 'use_true_curr_bspn=False', 'use_true_curr_aspn=False', 'use_all_previous_context=True'])

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

#/workspace/cvlachos/projects/UBAR/UBAR-MultiWOZ/data/augmentations/UBAR_10/UBAR_10_x2/roberta_replace
with open("./data/augmentations/UBAR_10/UBAR_10_x2/roberta_replace/data_for_damd2.json", "w") as outfile:
    outfile.write(data_json)

data = json.loads(open('./data/original/extra_data_for_damd.json', 'r', encoding='utf-8').read().lower())


logging.info('Running augmentation')
while k < len(data) :

  try :
    for n, i in enumerate(list(data.keys())[k:]):
      #print(k)
      #print(n,i)
      prev = 'user : '


      if i in data10 :
        print(i)
        for turn_num, turn in enumerate(data[i]['log']):
          next = []
          print(turn_num)
          temp = copy.deepcopy(turn['user'])

          for idi, ii in enumerate(data[i]['log']) :
              if idi >= turn_num :
                next.append(' user : ' + data[i]['log'][idi]['user'])
                next.append(' system : ' + data[i]['log'][idi]['resp_pre_delex'])
          next = ' '.join(next[1:])

          print(turn['user'], turn['user_delex'])
          turn['user'], turn['user_delex'] = roberta_paraphrase(turn['user'], turn['user_delex'],turn['cons_delex'], True, turn_num, i, prev, next)
          print(turn['user'], turn['user_delex'])
          #sdfgs()


          prev += (temp + ' system : ').strip()

          next = []
          for idi, ii in enumerate(data[i]['log']) :
              if idi > turn_num :
                next.append(' user : ' + data[i]['log'][idi]['user'])
                next.append(' system : ' + data[i]['log'][idi]['resp_pre_delex'])
          next = ' '.join(next[:])

          temp = copy.deepcopy(turn['resp_pre_delex'])
          print()
          print(turn['resp_pre_delex'], turn['resp'])
          turn['resp_pre_delex'], turn['resp'] = roberta_paraphrase(turn['resp_pre_delex'], turn['resp'],turn['cons_delex'] ,  False, turn_num, i, prev, next)
          print(turn['resp_pre_delex'], turn['resp'])

          prev += (temp + ' user : ').strip()

        #asdfbshjdfk()

        
        logging.info('\nSaving\n')

        data_aug = json.loads(open('./data/augmentations/UBAR_10/UBAR_10_x2/roberta_replace/data_for_damd2.json', 'r', encoding='utf-8').read().lower())
        data_aug[i] = data[i]


        with open("./data/augmentations/UBAR_10/UBAR_10_x2/roberta_replace/data_for_damd2.json", "w") as f:
          json.dump(data_aug, f, indent=2)
          count += 1
          logging.info('Have saved', count)

      k+=1

  except Exception as e :
    print('Something bad happened. Saving and waiting.',e)

  finally :
    break

print('Done')
