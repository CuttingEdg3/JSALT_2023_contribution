import os
import json
import pandas as pd
import argparse
from datasets import Dataset, load_from_disk


def normalize_dsct(glob_path, save=True, save_path=None):
  def get_turns(dialog):

    turns = []
    for dialog_info in dialog['turns']:
      turn = {}
      #print(dialog_info)


      turn['speaker_id'] = dialog_info['speaker_role'] + '_' + dialog['dialogue_id']
      turn['speaker_role'] = dialog_info['speaker_role']
      turn['turn_id'] = dialog_info['turn_id']
      turn['turn_string'] = dialog_info['utterance']
      turn['intent'] = dialog_info['intents']

      turn['voice_track'] = None
      turn['misc'] = {}

      turns.append(turn)
      #print()
    return turns

  df = pd.DataFrame(columns=['dataset','domains','data_split', 'dialog_id', 'turns', 'goals', 'misc'])

  for dset in ['development','test-banking', 'test-finance'] :
    print('Formating', dset)
    count = 1
    split = 'train'
    domain = 'insurance'

    if dset != 'development':
      split = 'test'
      domain = dset.split('-')[1]

    path = glob_path + '/{}/dialogues.jsonl'.format(dset)

    with open(path, 'r') as json_file:
        json_list = list(json_file)

    all_dialogs = []
    for json_str in json_list:
        dialog = json.loads(json_str)
        all_dialogs.append(dialog)

    for n, dialog in enumerate(all_dialogs):
      print('Working on', dset, 'item no',count,':', dialog['dialogue_id'])
      count += 1

      turns = get_turns(dialog)

      df.loc[len(df.index)] = ['DSTC11 Track2', domain , split, dialog['dialogue_id'], turns, {}, {}]

  df = Dataset.from_pandas(df)

  if save:
    save_df(df, save_path)
  return df

def save_df(df, path=None):
  """Saves DSTC11 into a pickle file as an HF DataFrame."""
  if not path:
    path = os.getcwd() + '/DSTC11_2_Normalized.hf'
  df.save_to_disk(path)
  print('Data saved to', path)

def load_hf_dstc11(path=None):
  """Loads DSTC11 as an HF Dataset."""
  if not path:
    path = os.getcwd() + '/DSTC11_2_Normalized.hf'

  df = load_from_disk(path)

  return df


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(dest='path',
                      help="Path of the DSTC track 2 directory where the development and 2 test sets are located (as downloaded from the original repo).")

  glob_path = str(parser.parse_args())

  args = parser.parse_args()
  glob_path = args.path

  df = normalize_dsct(glob_path, save=True)

  #df = load_hf_dstc11()



