import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import re
import pandas as pd
from features.building_features import clean_regex
import nltk
# from nltk import tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
np.random.seed(0)


def split(data, data_features, return_features, dev_ratio=0.3):
    train = data[data.type == 'training']
    train_features = data_features[train.index, :, :] if return_features else []
    train = shuffle(train)
    train = train.reset_index(drop=True).reset_index()
    # del train['id']
    # train = train.rename(columns={'index': 'id'})

    test = data[data.type == 'test']
    test_features = data_features[test.index, :, :] if return_features else []
    test = shuffle(test)
    test = test.reset_index(drop=True).reset_index()
    # del test['id']
    # test = test.rename(columns={'index': 'id'})

    self_train = {}
    self_dev = {}
    self_test = {}

    msk_dev = np.random.rand(len(train)) < dev_ratio
    self_dev['text'] = train['content'][msk_dev]
    self_dev['features'] = train_features[msk_dev, :, :] if return_features else []
    self_dev['label'] = train['label'][msk_dev]
    train = train[~msk_dev]
    self_train['text'] = train['content']
    self_train['features'] = train_features[~msk_dev, :, :] if return_features else []
    self_train['label'] = train['label']

    self_test['text'] = test['content']
    self_test['features'] = test_features if return_features else []
    self_test['label'] = test['label']
    return self_train, self_dev, self_test

## for splitting the Pubhealth datasset
def splithealths(train,valid,test,train_features,valid_features,test_features, return_features):

    self_train = {}
    self_dev = {}
    self_test = {}

    # msk_dev = np.random.rand(len(train)) < dev_ratio
    self_dev['text'] = train['content']
    self_dev['features'] = train_features if return_features else []
    self_dev['label'] = train['label']
    # train = train[~msk_dev]
    self_train['text'] = valid['content']
    self_train['features'] = valid_features if return_features else []
    self_train['label'] = valid['label']

    self_test['text'] = test['content']
    self_test['features'] = test_features if return_features else []
    self_test['label'] = test['label']
    return self_train, self_dev, self_test


#custom cleaning
# def clean_regex(text, keep_dot=False, split_text=False):
#     try:
#         text = re.sub(r'((http|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=;%&:/~+#-]*[\w@?^=%&;:/~+#-])?)', ' ',
#                       text)
#         text = re.sub(r'[^ ]+\.com', ' ', text)
#         text = re.sub(r'(\d{1,},)?\d{1,}(\.\d{1,})?', '', text)
#
#         text = re.sub(r'’', '\'', text)
#         text = re.sub(r'[^A-Za-z\'. ]', ' ', text)
#         text = re.sub(r'\.', '. ', text)
#         text = re.sub(r'\s{2,}', ' ', text)
#
#         text = re.sub(r'(\.\s)+', '.', str(text).strip())
#         text = re.sub(r'\.{2,}', '.', str(text).strip())
#         text = re.sub(r'(?<!\w)([A-Z])\.', r'\1', text)
#
#         text = re.sub(r'\'(?!\w{1,2}\s)', ' ', text)
#
#         text = text.split('.')
#         if keep_dot:
#             text = ' '.join([sent.strip() + ' . ' for sent in text])
#         else:
#             text = ' '.join([sent.strip() for sent in text])
#
#         text = text.lower()
#         return text.split() if split_text else text
#     except:
#         text = 'empty text'
#         return text.split() if split_text else text

def preprocessing(dataset,dev_ratio=0.2):

    if dataset=="MultiSourceFake":
        # dataMSF = pd.read_csv('./data/{}/sample.csv'.format(dataset))
        dataMSF = pd.read_csv('./data/{}/{}.csv'.format(dataset,dataset))
        # dataMSF = pd.read_csv(path+'/{}/{}.csv'.format(dataset, dataset))
        # text cleaning
        dataMSF['content'] = dataMSF["content"].apply(clean_regex)
        return dataMSF

    #new path with form = ./data/direc/
    if dataset == "ReCOVery":
        # preprocess files:rename,drop,gather text
        # tabrecovery = pd.read_csv(path+'{}/dataset/recovery-news-data.csv'.format(dataset))
        tabrecovery = pd.read_csv('./data/{}/dataset/recovery-news-data.csv'.format(dataset))

        tabrecovery = tabrecovery.rename({'reliability': 'label'}, axis=1)
        tabrecovery = tabrecovery[["title", "body_text", "label","type"]]
        tabrecovery['title'] = tabrecovery['title'].fillna(' ')
        # apply preprocessing for each raw
        tabrecovery['data_cleaned'] = tabrecovery['title'] + tabrecovery['body_text']
        tabrecovery['content'] = tabrecovery["data_cleaned"].apply(clean_regex)
        tabrecovery = tabrecovery.drop(columns=['title', 'body_text',"data_cleaned"])

        return tabrecovery
