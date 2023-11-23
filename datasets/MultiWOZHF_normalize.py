
import os
import pandas as pd
from datasets import load_dataset
from datasets import Dataset, load_from_disk


def normalize_multiwoz(save=False, save_path=None):
    """Normalizes MultiWOZ 2.2 into a Pandas DataFrame with the unified format."""

    multiwoz_dset = load_dataset("multi_woz_v22")

    def get_actions(dialog):

        acts = []
        for i in range(len(dialog['turns']['turn_id'])):
            turn = {}
            turn['speaker_id'] = 'USER' + '_' + dialog['dialogue_id'] if dialog['turns']['speaker'][
                                                                             i] == 0 else 'SYSTEM' + '_' + dialog[
                'dialogue_id']
            turn['speaker_role'] = 'USER' if dialog['turns']['speaker'][i] == 0 else 'SYSTEM'
            turn['turn_id'] = dialog['turns']['turn_id'][i]

            turn['action'] = dialog['turns']['dialogue_acts'][i]['dialog_act']['act_type']

            acts.append(turn)

        return acts

    def get_turns(dialog):

        turns = []
        # print(dialog)

        for i in range(len(dialog['turns']['turn_id'])):
            turn = {}

            turn['speaker_id'] = 'USER' + '_' + dialog['dialogue_id'] if dialog['turns']['speaker'][
                                                                             i] == 0 else 'SYSTEM' + '_' + dialog[
                'dialogue_id']
            turn['speaker_role'] = 'USER' if dialog['turns']['speaker'][i] == 0 else 'SYSTEM'
            turn['turn_id'] = dialog['turns']['turn_id'][i]
            turn['turn_string'] = dialog['turns']['utterance'][i]
            turn['intent'] = dialog['turns']['frames'][i]
            turn['intent'] = []

            for intent in (dialog['turns']['frames'][i]['state']):
                turn['intent'].append(intent['active_intent'])

            turn['voice_track'] = None
            turn['misc'] = {}

            turns.append(turn)

        return turns

    df = pd.DataFrame(columns=['dataset', 'domains', 'data_split', 'dialog_id', 'turns', 'goals', 'misc'])

    for dset in ['train', 'validation', 'test']:
        print('Formating {} set'.format(dset))
        count = 1
        for dialog in multiwoz_dset[dset]:
            print('Working on', dset, 'item no', count, ':', dialog['dialogue_id'])
            count += 1

            turns = get_turns(dialog)
            acts = get_actions(dialog)

            df.loc[len(df.index)] = ['MultiWOZ_2.2', dialog['services'], dset if dset != 'validation' else 'dev',
                                     dialog['dialogue_id'], turns, acts, {}]
    df = Dataset.from_pandas(df)

    if save:
        save_df(df, save_path)
    return df


def save_df(df, path=None):
    """Saves MultiWOZ into a pickle file as a Pandas DataFrame."""
    if not path:
        path = os.getcwd() + '/WOZ2HFNormalized.hf'
    df.save_to_disk(path)
    print('Data saved to', path)


def load_hf_multiwoz(path=None):
    """Loads MultiWOZ as an HF Dataset."""
    if not path:
        path = os.getcwd() + '/WOZ2HFNormalized.hf'

    df = load_from_disk(path)

    return df


if __name__ == '__main__':

    df = normalize_multiwoz(save=True)

    # df = load_hf_multiwoz()
