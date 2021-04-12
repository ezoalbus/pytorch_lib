import os
from glob import glob
import pandas as pd
import linecache


def prepare_data(to_dump=True):
    data_dir = './data/'
    text_dir = './data/text/'

    if os.path.exists(data_dir + 'datasets.pkl'):
        return pd.read_pickle(data_dir + 'datasets.pkl')

    # カテゴリを取得
    categories = []
    for name in os.listdir(text_dir):
        if os.path.isdir(text_dir + name):
            categories.append(name)

    datasets = pd.DataFrame(columns=['title', 'category'])
    for cat in categories:
        path = text_dir + cat + '/*.txt'
        files = glob(path)
        for text_name in files:
            title = linecache.getline(text_name, 3)  # 3行目を取得
            s = pd.Series([title, cat], index=datasets.columns)
            datasets = datasets.append(s, ignore_index=True)

    # データフレームシャッフル
    datasets = datasets.sample(frac=1).reset_index(drop=True)

    datasets.to_pickle(data_dir + 'datasets.pkl')

    return datasets
    