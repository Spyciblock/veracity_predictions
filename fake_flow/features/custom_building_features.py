import warnings
warnings.filterwarnings("ignore")

import re
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
from tqdm import tqdm
from os.path import exists
from os.path import join
from joblib import Parallel, delayed

from features.emotional.loading_emotional_lexicons import emotional_lexicons
from features.sentiment.loading_sentiment_lexicons import sentiment_lexicons
from features.morality.morality import MORALITY_class
from features.imageability.imageability import imageability_class
from features.hyperbolic.hyperbolic import hyperbolic_class

# config
np.random.seed(0)
tqdm.pandas()


def clean_regex(text, keep_dot=False, split_text=False):
    try:

        contraction_dict = {"ain't": "is not", "aren't": "are not", "can't": "cannot", "'cause": "because",
                                "could've": "could have", "couldn't": "could not", "didn't": "did not",
                                "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not",
                                "haven't": "have not", "he'd": "he would", "he'll": "he will", "he's": "he is",
                                "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
                                "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have",
                                "I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have",
                                "i'll": "i will", "i'll've": "i will have", "i'm": "i am", "i've": "i have",
                                "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will",
                                "it'll've": "it will have", "it's": "it is", "let's": "let us", "ma'am": "madam",
                                "mayn't": "may not", "might've": "might have", "mightn't": "might not",
                                "mightn't've": "might not have", "must've": "must have", "mustn't": "must not",
                                "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have",
                                "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have",
                                "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
                                "she'd": "she would", "she'd've": "she would have", "she'll": "she will",
                                "she'll've": "she will have", "she's": "she is", "should've": "should have",
                                "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have",
                                "so's": "so as", "this's": "this is", "that'd": "that would",
                                "that'd've": "that would have", "that's": "that is", "there'd": "there would",
                                "there'd've": "there would have", "there's": "there is", "here's": "here is",
                                "they'd": "they would", "they'd've": "they would have", "they'll": "they will",
                                "they'll've": "they will have", "they're": "they are", "they've": "they have",
                                "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have",
                                "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have",
                                "weren't": "were not", "what'll": "what will", "what'll've": "what will have",
                                "what're": "what are", "what's": "what is", "what've": "what have", "when's": "when is",
                                "when've": "when have", "where'd": "where did", "where's": "where is",
                                "where've": "where have", "who'll": "who will", "who'll've": "who will have",
                                "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have",
                                "will've": "will have", "won't": "will not", "won't've": "will not have",
                                "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have",
                                "y'all": "you all", "y'all'd": "you all would", "y'all'd've": "you all would have",
                                "y'all're": "you all are", "y'all've": "you all have", "you'd": "you would",
                                "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
                                "you're": "you are", "you've": "you have"}

        def replace_contractions(text, contraction_dict):
          contractions_re = re.compile('(%s)' % '|'.join(contraction_dict.keys()))

          def replace(match):
             return contraction_dict[match.group(0)]

          return contractions_re.sub(replace, text)
        # replace contractions
        text = replace_contractions(text, contraction_dict)
        print("New Preprocessing added")

        text = re.sub(r'((http|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=;%&:/~+#-]*[\w@?^=%&;:/~+#-])?)', ' ',
                      text)
        text = re.sub(r'[^ ]+\.com', ' ', text)
        text = re.sub(r'(\d{1,},)?\d{1,}(\.\d{1,})?', '', text)
        text = re.sub(r'’', '\'', text)
        text = re.sub(r'[^A-Za-z\'. ]', ' ', text)
        text = re.sub(r'\.', '. ', text)
        text = re.sub(r'\s{2,}', ' ', text)

        text = re.sub(r'(\.\s)+', '.', str(text).strip())
        text = re.sub(r'\.{2,}', '.', str(text).strip())
        text = re.sub(r'(?<!\w)([A-Z])\.', r'\1', text)

        text = re.sub(r'\'(?!\w{1,2}\s)', ' ', text)

        text = text.split('.')
        if keep_dot:
            text = ' '.join([sent.strip() + ' . ' for sent in text])
        else:
            text = ' '.join([sent.strip() for sent in text])

        text = text.lower()
        return text.split() if split_text else text
    except:
        text = 'empty text'
        return text.split() if split_text else text


class append_split_3D(BaseEstimator, TransformerMixin):
    def __init__(self, segments_number=20, max_len=50, mode='append'):
        self.segments_number = segments_number
        self.max_len = max_len
        self.mode = mode
        self.appending_value = -5.123

    def fit(self, X, y=None):
        return self

    def transform(self, data):
        if self.mode == 'append':
            self.max_len = self.max_len - data.shape[2]
            appending = np.full((data.shape[0], data.shape[1], self.max_len), self.appending_value)
            new = np.concatenate([data, appending], axis=2)
            return new
        elif self.mode == 'split':
            tmp = []
            for item in range(0, data.shape[1], self.segments_number):
                tmp.append(data[:, item:(item + self.segments_number), :])
            tmp = [item[item != self.appending_value].reshape(data.shape[0], self.segments_number, -1) for item in tmp]
            new = np.concatenate(tmp, axis=2)
            return new
        else:
            print('Error: Mode value is not defined')
            exit(1)

class segmentation(BaseEstimator, TransformerMixin):

    def __init__(self, n_jobs=1, segments_number=20):
        self.n_jobs = n_jobs
        self.segments_number = segments_number

    def fit(self, X, y=None):
        return self

    def transform(self, data):
        out = []
        for sentence in data:
            tmp = np.array_split(sentence, self.segments_number)
            tmp = [np.sum(item, axis=0) / sentence.shape[0] for item in tmp]
            out.append(tmp)
        out = np.array(out)
        return out

class segmentation_text(BaseEstimator, TransformerMixin):

    def __init__(self, n_jobs=1, segments_number=20):
        self.n_jobs = n_jobs
        self.segments_number = segments_number

    def fit(self, X, y=None):
        return self

    def transform(self, data):
        data = Parallel(n_jobs=1, backend="multiprocessing", prefer="processes") \
            (delayed(clean_regex)(sentence, keep_dot=False, split_text=True) for sentence in tqdm(data, desc='Text Segmentation'))
        if isinstance(data, list):
            data = np.array([np.array(sent) for sent in data])
        out = []
        for sentence in data:
            try:
                tmp = np.array_split(sentence, self.segments_number)
                tmp = ' . '.join([' '.join(item.tolist()) for item in tmp])
            except:
                print()
            out.append(tmp)
        return out


class emotional_features(BaseEstimator, TransformerMixin):

    def __init__(self, path='', n_jobs=1, model_name='', representation='frequency'):
        self.path = path
        self.n_jobs = n_jobs
        self.model_name = model_name
        self.representation = representation

    def fit(self, X, y=None):
        return self

    def error_representation(self):
        print('\n\nError: check the value of the variable "representation".')
        exit(1)

    def transform(self, data):
        #new
        file_name = './processed_files/features/emotional_features_preproc_{}_{}.npy'.format(self.model_name, self.representation)
        if exists(file_name):
            features = np.load(file_name,allow_pickle=True).tolist()
        else:
            #new

            data = Parallel(n_jobs=self.n_jobs, backend="multiprocessing", prefer="processes") \
                    (delayed(clean_regex)(sentence, False, True) for sentence in tqdm(data, desc='Cleaning text'))

            emo = emotional_lexicons(path=join(self.path, 'emotional'))
            loop = tqdm(data)
            loop.set_description('Building emotional_features ({})'.format(self.representation))

            features = Parallel(n_jobs=self.n_jobs, backend="multiprocessing", prefer="processes") \
                (delayed(emo.frequency if self.representation == 'frequency' else emo.intensity if self.representation == 'intensity' else self.error_representation())
                 (sentence) for sentence in loop)

            features = [np.array(item) for item in features]
            np.save(file_name, features)
        return features

class sentiment_features(BaseEstimator, TransformerMixin):

    def __init__(self, path='', n_jobs=1, model_name=''):
        self.path = path
        self.n_jobs = n_jobs
        self.model_name = model_name
    def fit(self, X, y=None):
        return self

    def transform(self, data):
        #new
        file_name = './processed_files/features/sentiment_features_preproc_{}.npy'.format(self.model_name)

        if exists(file_name):
            features = np.load(file_name,allow_pickle=True).tolist()
        else:
            data = Parallel(n_jobs=self.n_jobs, backend="multiprocessing", prefer="processes") \
                    (delayed(clean_regex)(sentence, False, True) for sentence in tqdm(data, desc='Cleaning text'))

            senti = sentiment_lexicons(path=join(self.path, 'sentiment'))
            loop = tqdm(data)
            loop.set_description('Building sentiment_features')

            features = Parallel(n_jobs=self.n_jobs, backend="multiprocessing", prefer="processes")(delayed(senti.score)(sentence) for sentence in loop)
            features = [np.array(item) for item in features]
            np.save(file_name, features)
        return features

class morality_features(BaseEstimator, TransformerMixin):

    def __init__(self, path='', n_jobs=1, model_name=''):
        #preproc=False
        self.path = path
        self.n_jobs = n_jobs
        self.model_name = model_name

    def fit(self, X, y=None):
        return self

    def transform(self, data):
        #new

        file_name = './processed_files/features/morality_features_preproc_{}.npy'.format(self.model_name)
        if exists(file_name):
            features = np.load(file_name,allow_pickle=True).tolist()
        else:
            #new
            data = Parallel(n_jobs=self.n_jobs, backend="multiprocessing", prefer="processes") \
                    (delayed(clean_regex)(sentence, False, True) for sentence in tqdm(data, desc='Cleaning text'))

            lex = MORALITY_class(path=join(self.path, 'morality'))
            loop = tqdm(data)
            loop.set_description('Building Morality_features')

            features = Parallel(n_jobs=self.n_jobs, backend="multiprocessing", prefer="processes")(delayed(lex.score)(sentence) for sentence in loop)
            features = [np.array(item) for item in features]
            np.save(file_name, features)
        return features

class imageability_features(BaseEstimator, TransformerMixin):

    def __init__(self, path='', n_jobs=1, model_name=''):
        self.path = path
        self.n_jobs = n_jobs
        self.model_name = model_name

    def fit(self, X, y=None):
        return self

    def transform(self, data):
        #new
        file_name = './processed_files/features/imageability_features_preproc_{}.npy'.format(self.model_name)

        if exists(file_name):
            features = np.load(file_name,allow_pickle=True).tolist()
        else:

            data = Parallel(n_jobs=self.n_jobs, backend="multiprocessing", prefer="processes") \
                    (delayed(clean_regex)(sentence, False, True) for sentence in tqdm(data, desc='Cleaning text'))

            lex = imageability_class(path=join(self.path, 'imageability'))
            loop = tqdm(data)
            loop.set_description('Building Imageability_features')

            features = Parallel(n_jobs=self.n_jobs, backend="multiprocessing", prefer="processes")(delayed(lex.score)(sentence) for sentence in loop)
            features = [np.array(item) for item in features]
            np.save(file_name, features)
        return features

class hyperbolic_features(BaseEstimator, TransformerMixin):

    def __init__(self, path='', n_jobs=1, model_name=''):
        self.path = path
        self.n_jobs = n_jobs
        self.model_name = model_name
    def fit(self, X, y=None):
        return self

    def transform(self, data):

        file_name = './processed_files/features/hyperbolic_features_preproc_{}.npy'.format(self.model_name)
        if exists(file_name):
            features = np.load(file_name,allow_pickle=True).tolist()
        else:
            #new

            data = Parallel(n_jobs=self.n_jobs, backend="multiprocessing", prefer="processes") \
                (delayed(clean_regex)(sentence, False, True) for sentence in tqdm(data, desc='Cleaning text'))

            lex = hyperbolic_class(path=join(self.path, 'hyperbolic'))
            loop = tqdm(data)
            loop.set_description('Building Hyperbolic_features')

            features = Parallel(n_jobs=self.n_jobs, backend="multiprocessing", prefer="processes")(delayed(lex.score)(sentence) for sentence in loop)
            features = [np.array(item) for item in features]
            np.save(file_name, features)
        return features


def manual_features(path='', n_jobs=1, model_name='', segments_number=20, emo_rep='frequency'):
    manual_feats = Pipeline([
        ('FeatureUnion', FeatureUnion([
            ('1', Pipeline([
                ('emotional_features', emotional_features(path=path, n_jobs=n_jobs, model_name=model_name, representation=emo_rep)), #preproc=newproc
                ('segmentation', segmentation(n_jobs=n_jobs, segments_number=segments_number)),
                ('append', append_split_3D(segments_number=segments_number, max_len=50, mode='append')),
            ])),
            ('2', Pipeline([
                ('sentiment_features', sentiment_features(path=path, n_jobs=n_jobs, model_name=model_name)),
                ('segmentation', segmentation(n_jobs=n_jobs, segments_number=segments_number)),
                ('append', append_split_3D(segments_number=segments_number, max_len=50, mode='append')),
            ])),
            ('3', Pipeline([
                ('morality_features', morality_features(path=path, n_jobs=n_jobs, model_name=model_name)),
                ('segmentation', segmentation(n_jobs=n_jobs, segments_number=segments_number)),
                ('append', append_split_3D(segments_number=segments_number, max_len=50, mode='append')),
            ])),
            ('4', Pipeline([
                ('imageability_features', imageability_features(path=path, n_jobs=n_jobs, model_name=model_name)),
                ('segmentation', segmentation(n_jobs=n_jobs, segments_number=segments_number)),
                ('append', append_split_3D(segments_number=segments_number, max_len=50, mode='append')),
            ])),
            ('5', Pipeline([
                ('hyperbolic_features', hyperbolic_features(path=path, n_jobs=n_jobs, model_name=model_name)),
                ('segmentation', segmentation(n_jobs=n_jobs, segments_number=segments_number)),
                ('append', append_split_3D(segments_number=segments_number, max_len=50, mode='append')),
            ])),
        ], n_jobs=1)),
        ('split', append_split_3D(segments_number=segments_number, max_len=50, mode='split'))
    ])
    return manual_feats


if __name__ == '__main__':
    df = pd.DataFrame([{'text': "I don't to xsdf"},
                       {'text': "she can want to be witt"}])
    res = manual_features(n_jobs=4).fit_transform(df)
    x = pd.Series(res.tolist())
    print('')
