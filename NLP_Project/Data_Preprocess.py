import pandas as pd
import json
import requests


def get_split_data(data):

    if data == 'full':
        df = json.loads(requests.get('https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json').text)

    elif data == 'small':
        df = json.loads(requests.get('https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_small.json').text)

    elif data == 'imbal':
        df = json.loads(requests.get('https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_imbalanced.json').text)

    else:
        df = json.loads(requests.get('https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_oos_plus.json').text)

    ''' Getting Training data '''
    inscope_train = pd.DataFrame(df['train'],columns=['query','intent'])
    oss_train = pd.DataFrame(df['oos_train'],columns=['query','intent'])

    ''' Getting Validation data '''
    inscope_val = pd.DataFrame(df['val'],columns=['query','intent'])
    oss_val = pd.DataFrame(df['oos_val'],columns=['query','intent'])

    ''' Getting Test data '''
    inscope_test = pd.DataFrame(df['test'],columns=['query','intent'])
    oss_test = pd.DataFrame(df['oos_test'],columns=['query','intent'])


    train_df = pd.concat([inscope_train,oss_train])
    val_df = pd.concat([inscope_val,oss_val])
    test_df = pd.concat([inscope_test,oss_test])



    # return inscope_train,inscope_val,inscope_test
    return train_df,val_df,test_df,inscope_train,inscope_val,inscope_test

def get_in_scope_data(data):

    train_df,val_df,test_df,inscope_train,inscope_val,inscope_test = get_split_data(data)
    return inscope_train,inscope_val,inscope_test

def get_oos_binary_preprocess(data):
    train_df,val_df,test_df,inscope_train,inscope_val,inscope_test = get_split_data(data)


    train_df.loc[train_df.intent != 'oos','intent'] = 0
    train_df.loc[train_df.intent == 'oos','intent'] = 1

    test_df.loc[test_df.intent != 'oos','intent'] = 0
    test_df.loc[test_df.intent == 'oos','intent'] = 1

    val_df.loc[val_df.intent != 'oos','intent'] = 0
    val_df.loc[val_df.intent == 'oos','intent'] = 1

    return train_df,val_df,test_df



if __name__ == '__main__':
    # _, val_df, _ = get_split_data('full')
    get_oos_binary_preprocess('full')







