import argparse
import json
import os
import pandas as pd
from datasets import Dataset, load_from_disk


#/Users/cvlachos/PycharmProjects/MultiWOZ_data/multiwoz-master/data/MultiWOZ_2.2

def normalize_multiwoz(glob_path, save=False, save_path=None):
    """Normalizes MultiWOZ 2.2 into a Pandas DataFrame with the unified format."""
    def get_actions(dialog, dialog_acts):
        acts = []
        for dialog_info in dialog['turns']:
            turn = {}
            turn['speaker_id'] = dialog_info['speaker'] + '_' + dialog['dialogue_id']
            turn['speaker_role'] = dialog_info['speaker']
            turn['turn_id'] = dialog_info['turn_id']

            turn['action'] = dialog_acts[dialog['dialogue_id']][dialog_info['turn_id']]['dialog_act']

            acts.append(turn)
        return acts
    def get_turns(dialog):
        turns = []
        for dialog_info in dialog['turns']:
            turn = {}

            turn['speaker_id'] = dialog_info['speaker'] + '_' + dialog['dialogue_id']
            turn['speaker_role'] = dialog_info['speaker']
            turn['turn_id'] = dialog_info['turn_id']
            turn['turn_string'] = dialog_info['utterance']
            turn['intent'] = []

            if turn['speaker_role'] == 'USER':
                for dialog_turn in dialog_info['frames']:
                    if dialog_turn['state']['active_intent'] != "NONE":
                        turn['intent'].append(dialog_turn['state']['active_intent'])

            turn['voice_track'] = None
            turn['misc'] = {}

            turns.append(turn)
        return turns
    df = pd.DataFrame(columns=['dataset', 'domains', 'data_split', 'dialog_id', 'turns', 'goals', 'misc'])
    dialog_acts = json.load(open(glob_path + '/dialog_acts.json'))

    for dset in ['train', 'dev', 'test']:
        print('Formatting {} set'.format(dset))
        count = 1

        for subdset in os.listdir(glob_path + '/'+ dset):
            if subdset != 'all_dialogs.json':
                dialog_json = json.load(open(glob_path + '/' + dset + '/' + subdset))

                for num, dialog in enumerate(dialog_json[:]):
                    print('Working on', dset, 'item no', count, ':', dialog['dialogue_id'])
                    count += 1
                    turns = get_turns(dialog)
                    goals = get_actions(dialog, dialog_acts)
                    df.loc[len(df.index)] = ['MultiWOZ_2.2', dialog['services'], dset, dialog['dialogue_id'], turns,
                                             goals, {}]

    df = Dataset.from_pandas(df)

    if save:
        save_df(df, save_path)
    return df


def save_df(df, path=None):
    """Saves MultiWOZ into a pickle file as a Pandas DataFrame."""
    if not path:
        path = os.getcwd() + '/WOZ2Normalized.hf'
    df.save_to_disk(path)
    print('Data saved to', path)



def load_hf_multiwoz(path=None):
    """Loads MultiWOZ as an HF Dataset."""
    if not path:
        path = os.getcwd() + '/WOZ2Normalized.hf'

    df = load_from_disk(path)

    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='path',
                        help="Path of the MultiWOZ_2.2 directory where the train, dev and test sets are located (as downloaded from the original repo).")

    args = parser.parse_args()
    glob_path = args.path

    df = normalize_multiwoz(glob_path, save=True)
