import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from features.custom_building_features  import manual_features, segmentation_text, clean_regex
from data.utils import split,lstmsplit,splithealths

## Install transformers, datasets before
from datasets import list_datasets, list_metrics, load_dataset, load_metric
np.random.seed(0)


class custom_preparation():
    def __init__(self,dataname,segments_number=10, n_jobs=-1, emo_rep='frequency',path="",model="Fakeflow", return_features=True,
                  text_segments=False, clean_text=True):
        self.n_jobs = n_jobs
        self.segments_number = segments_number
        self.emo_rep = emo_rep
        self.path = path
        self.model = model
        self.return_features = return_features
        self.text_segments = text_segments
        self.clean_text = clean_text

    ## use only Pubhealths dataset

    def custom_features_plit(self,dataname,content):
        content_features = []
        """Extract features, segment text, clean it."""
        if self.return_features:
            content_features = manual_features(n_jobs=self.n_jobs, path='./features', model_name=dataname,segments_number=self.segments_number, emo_rep=self.emo_rep).transform(content['content'])

        """In segmentation we already clean the text to keep the DOTS (.) only."""
        if self.text_segments:
            content['content'] = segmentation_text(segments_number=self.segments_number).transform(content['content'])
        elif self.clean_text:
            content['content'] = content['content'].map(lambda text: clean_regex(text, keep_dot=True))

        return content,content_features

    #MultiSourceFake
    def custom_prepare_input(self,dataname='PubHealth'):
        ##path with ./data/
        # if dataname == "MultiSourceFake":
        #     content = pd.read_csv('./data/{}/sample.csv'.format(dataname))

        #new path with form = ./data/direc/
        if dataname == "ReCOVery":
            #preprocess files:rename,drop,gather text  recovery-news-data
            content = pd.read_csv('./data/{}/dataset/ReCOVery.csv'.format(dataname))
            content = content.rename({'reliability': 'label'}, axis=1)
            content = content[["title", "body_text", "label","type"]]
            content['title'] = content['title'].fillna(' ')
            content['content'] = content['title'] + content['body_text']
            content = content.drop(columns=['title', 'body_text'])

        if dataname =="PubHealth":
            # preprocess puhealth

            dataset = load_dataset("health_fact")
            # split data and reformat into a dataframe
            train_data = pd.DataFrame.from_dict(dataset["train"])
            valid_data = pd.DataFrame.from_dict(dataset["validation"])
            test_data = pd.DataFrame.from_dict(dataset["test"])
            # print(train_data.info())
            # print(valid_data.info())
            # print(test_data.info())

            # remove label -1
            train_data = train_data[train_data['label'] != -1]
            valid_data = valid_data[valid_data['label'] != -1]
            test_data = test_data[test_data['label'] != -1]

            # change type in  columns
            data_types = {'claim': 'str', 'explanation': 'str', 'fact_checkers': 'str', 'main_text': 'str',
                          'sources': 'str', 'subjects': 'str',
                          'claim_id': 'int', 'label': 'int'}
            train_data = train_data.astype(data_types)
            valid_data = valid_data.astype(data_types)
            test_data = test_data.astype(data_types)

            # rename column claim to content
            train_data = train_data.rename({'claim': 'content'}, axis=1)
            valid_data = valid_data.rename({'claim': 'content'}, axis=1)
            test_data = test_data.rename({'claim': 'content'}, axis=1)
            # train_data = train_data[["content", "main_text", "explanation", "label"]]

            ## group text or not, then drop columns

            # train_data['content'] = train_data['content'] + train_data['main_text']
            # valid_data['content'] = valid_data['content'] + valid_data['main_text']
            # test_data['content'] = test_data['content'] + test_data['main_text']

            train_data = train_data.drop(columns=['fact_checkers', 'sources','subjects'])
            valid_data = valid_data.drop(columns=['fact_checkers', 'sources', 'subjects'])
            test_data = test_data.drop(columns=['fact_checkers', 'sources', 'subjects'])

            # get features and splits
            train_df,train_features = custom_preparation.custom_features_plit(self,dataname="PubHealth_training",content=train_data)
            valid_df, valid_features = custom_preparation.custom_features_plit(self,dataname="PubHealth_validation",content= valid_data)
            test_df, test_features = custom_preparation.custom_features_plit(self,dataname="PubHealth_test",content= test_data)
            print("train_df shape=", train_df.shape)
            print("train_df features shape=",train_features.shape)
            train, dev, test = splithealths(train_df,valid_df,test_df,train_features,valid_features,test_features, self.return_features)

        else:
            content_features = []
            """Extract features, segment text, clean it."""
            if self.return_features:
                content_features = manual_features(n_jobs=self.n_jobs, path='./features', model_name=dataname,segments_number=self.segments_number, emo_rep=self.emo_rep).transform(content['content'])

            """In segmentation we already clean the text to keep the DOTS (.) only."""
            if self.text_segments:
                content['content'] = segmentation_text(segments_number=self.segments_number).transform(content['content'])
            elif self.clean_text:
                content['content'] = content['content'].map(lambda text: clean_regex(text, keep_dot=True))

            train, dev, test = split(content, content_features, self.return_features)

        return train, dev, test

if __name__ == '__main__':

    example = custom_preparation(dataname="PubHealth")
    train, dev, test = example.custom_prepare_input(dataname='PubHealth')
    print("shape train=",train)
    print("shape dev=", dev)
    print("shape test=", test)
