
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sms
import os
from collections import Counter
# import webbrowser
# import graphviz
from sklearn.preprocessing import StandardScaler,MinMaxScaler,OneHotEncoder

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer

# from sklearn.model_selection import PredefinedSplit
from sklearn.pipeline import Pipeline

from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score,classification_report
from argparse import ArgumentParser
from nltk.tokenize.treebank import TreebankWordTokenizer,TreebankWordDetokenizer
# Use BERT model wit simpletransformers
from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

import json
import tqdm
import torch
import torch.nn as nn
import transformers
import datetime

# Load the TensorBoard notebook extension
# import  tensorflow as tf
# %load_ext tensorboard
# import datetime, os

# import utils
# import torch_glove
# import torch_rnn_classifier,torch_model_base
# import torch_shallow_neural_classifier
from nltk.corpus import stopwords
from nltk.tokenize.treebank import TreebankWordTokenizer
import nltk
nltk.download("stopwords")


# def treebank_tokenize(s):
#     treebanktokenizer = TreebankWordTokenizer()
#     return treebanktokenizer.tokenize(s)
#
# def preprocess(x):
#   treetokens = TreebankWordTokenizer()
#   tokens=treetokens.tokenize(x)
#   filtered_words = [word for word in tokens if word not in stopwords.words('english')]
#   # print("spli=",filtered_words)
#   # print("spli2=",tokens[:50])
#   return " ".join(filtered_words)

def preprocesser(train_data,valid_data,test_data):

    # remove label -1
    train_data = train_data[train_data['label'] != -1]
    valid_data = valid_data[valid_data['label'] != -1]
    test_data = test_data[test_data['label'] != -1]

    # change type in  columns
    data_types = {'claim': 'string', 'explanation': 'string', 'fact_checkers': 'string', 'main_text': 'string',
                  'sources': 'string', 'subjects': 'string',
                  'claim_id': 'int', 'label': 'int'}
    # data_types = {'claim': 'str', 'explanation': 'str', 'fact_checkers': 'str', 'main_text': 'str',
    #                           'sources': 'str', 'subjects': 'str',
    #                           'claim_id': 'int', 'label': 'int'}
    train_data = train_data.astype(data_types)
    valid_data = valid_data.astype(data_types)
    test_data = test_data.astype(data_types)

    # rename column claim to text and drop columns
    train_data = train_data.rename({'claim': 'text', 'label': 'labels'}, axis=1)
    valid_data = valid_data.rename({'claim': 'text', 'label': 'labels'}, axis=1)
    test_data = test_data.rename({'claim': 'text', 'label': 'labels'}, axis=1)
    # train_data = train_data[["content", "main_text", "explanation", "label"]]

    # group claim and main_text
    # train_data['content'] = train_data['claim'] + train_data['main_text']
    # valid_data['content'] = valid_data['claim'] + valid_data['main_text']
    # test_data['content'] = test_data['claim'] + test_data['main_text']

    train_data = train_data.drop(columns=['fact_checkers', 'sources', 'date_published'])
    valid_data = valid_data.drop(columns=['fact_checkers', 'sources', 'date_published'])
    test_data = test_data.drop(columns=['fact_checkers', 'sources', 'date_published'])
    return train_data,valid_data,test_data


def save(path,result,name):
  import json
  # create json object from dictionary
  json = json.dumps(result)
  # open file for writing, "w"
  f = open(path+str(name)+".json","w")
  # write json object to file
  f.write(json)
  # close file
  f.close()


if __name__ == '__main__':
    ### Import Dataset
    # Let's import the library.
    from datasets import list_datasets, list_metrics, load_dataset, load_metric

    parser = ArgumentParser()
    parser.add_argument("-model", "--model", help="Model name", type=str,default="roberta")
    parser.add_argument("-modelversion", "--modelversion", help="Version of model",default="roberta-base", type=int)
    parser.add_argument("-path", "--outputpath", help="Path for output", default="", type=str)

    args = parser.parse_args()
    model=args.model
    modelname=args.modelversion
    mypath =args.outputpath
    if mypath=="":
        path = '/content/drive/My Drive/Colab Notebooks/Onclusive_work/'
    else:
        path = os.getcwd()+str(mypath)

    dataset = load_dataset("health_fact")
    #split data and reformat in dataframe
    train_data = pd.DataFrame.from_dict(dataset["train"])
    valid_data = pd.DataFrame.from_dict(dataset["validation"])
    test_data = pd.DataFrame.from_dict(dataset["test"])
    # print(train_data.info())
    #preprocessing
    train_data,valid_data,test_data = preprocesser(train_data,valid_data,test_data)

    # model="bert"
    # modelname= "bert-base-multilingual-cased" #" bert-base-uncased"
    # model = "roberta"
    # modelname = "roberta-base"

    # Use  model withsimpletransformers
    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)
    # Preparing train data
    train_df = train_data[['text', 'labels']]
    valid_df = valid_data[['text', 'labels']]
    test_df = test_data[['text', 'labels']]

    # Optional model configuration
    model_args = ClassificationArgs(num_train_epochs=5, train_batch_size=64,
                                    output_dir="outputTransformer", overwrite_output_dir=True,
                                    use_early_stopping=True)

    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    device = torch.device('cuda:0')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Create a ClassificationModel num_labels=4, use_cuda=device,cuda_device=0
    model = ClassificationModel(
        model, modelname, args=model_args, use_cuda=False, num_labels=4
    )

    # Train the model
    model.train_model(train_df, acc=classification_report)
    # Evaluate the model
    result, model_outputs, wrong_predictions = model.eval_model(valid_df, acc=classification_report)
    print("************DEV SET***********")
    print("result=", result)
    print("model_outputs=", model_outputs)
    print("wrong=", wrong_predictions)
    # Make predictions with the model
    # predictions, raw_outputs = model.predict(test_data["claim"])

    save(path, result,name=str(model)+"_"+str(modelname)+'_results')

    # model.convert_to_onnx(path+"onnx_outputs")
    predictions, raw_outputs, wrong2 = model.eval_model(test_df, acc=classification_report)
    print("************TEST SET***********")
    print("result=", predictions)
    print("model_outputs=", raw_outputs)
    print("wrong=", wrong2)
    save(path, predictions, name=str(model)+"_"+str(modelname)+"_preds")