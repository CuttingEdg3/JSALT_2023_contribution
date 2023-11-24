import stanza
stanza.download('en')
nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse')

import argparse
import os
import json
import logging
import string

# from google.colab import drive
# drive.mount('/content/drive')


import crop_rotate_augment.augment





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



def conllufy(input_pre_delex, input_init, delex_dict, op):

  doc = nlp(input_pre_delex)
  st_all = ''


  for k, sent in enumerate(doc.sentences):
      st = ''
      for n, word in enumerate(sent.words):
          st += f'{word.id}\t{word.text}\t_\t{word.upos}\t_\t_\t{word.head}\t{word.deprel}\t{word.head}:{word.deprel}\t_'+'\n'


      sents = crop_rotate_augment.augment.augment(st[:-1], op, 3)
      begin = 0

      if len(sents) > 1 :
        begin = 1

      for i in sents[begin:]:
        ret = ''
        for j in i.split('\n')[:-2]:
          ret+=j.split('\t')[1] + ' '
        st_all += ' '+ret
        if len(sents) > 1 :
          st_all += '.'
        break

  st_ret = st_all
  for i in delex_dict:
    if i not in st_ret:
      print('Sentence missing :',i)
      return input_pre_delex, input_init

    st_ret = st_ret.replace(i, delex_dict[i])

  if len(st_all.split()) < len(input_pre_delex.split())/2 :
    return input_pre_delex, input_init

  return st_all.strip(), st_ret.strip()


def get_delex_dict(text_splt, delex_splt):
  delex_tokens = {}
  
  try : 
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
  except :
      logging.info('Error in delex')
  return delex_tokens






#os.chdir('/content/drive/MyDrive/PhD/UBAR/UBAR-MultiWOZ')
logging.info('Reading data')
data10 = []

data10_pre = []


with open('./10_percent_training.txt' , 'r') as f:
  lines = f.readlines()

  for line in lines:
    data10.append(line.strip())

with open('./data/original/10_percent_training.txt' , 'r') as f:
  lines = f.readlines()

  for line in lines:
    data10_pre.append(line.strip())


k = 0
count = 0
data_json = json.dumps({})


with open("./data/augmentations/crop_rotate/data_for_damd.json", "w") as outfile:
    outfile.write(data_json)

data = json.loads(open('./data/original/extra_data_for_damd.json', 'r', encoding='utf-8').read().lower())


logging.info('Running augmentation')

for n, i in enumerate(list(data.keys())[k:]):
  #print(k)
  #print(n,i)

  if i in data10 and i not in data10_pre:
    print(i)
    for turn_num, turn in enumerate(data[i]['log']):
      print(turn_num)

      print(turn['user'], turn['user_delex'])
      turn['user'], turn['user_delex'] = conllufy(turn['user'], turn['user_delex'], get_delex_dict(turn['user'].split(), turn['user_delex'].split()), 'rotate')
      print(turn['user'], turn['user_delex'])



      print()
      print(turn['resp_pre_delex'], turn['resp'])
      turn['resp_pre_delex'], turn['resp'] = conllufy(turn['resp_pre_delex'], turn['resp'], get_delex_dict(turn['resp_pre_delex'].split(), turn['resp'].split()), 'rotate')
      print(turn['resp_pre_delex'], turn['resp'])

      print()


    logging.info('Saving')

    data_aug = json.loads(open('./data/augmentations/crop_rotate/data_for_damd.json', 'r', encoding='utf-8').read().lower())
    data_aug[i] = data[i]


    with open("./data/augmentations/crop_rotate/data_for_damd.json", "w") as f:
      json.dump(data_aug, f, indent=2)
      count += 1
      logging.info('Have saved', count)

  k+=1




print('Done')
