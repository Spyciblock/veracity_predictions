import os,glob
from os.path import exists, join
import numpy as np
from tqdm import tqdm
import pandas as pd
from argparse import ArgumentParser
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# extract text from files in directory and get two dataframes


def prepare_recovery(path,savepath,args):
    #read
    tabrecovery = pd.read_csv(path)
    #filter
    true_tabrecovery = tabrecovery.loc[tabrecovery["reliability"] == 1]
    fake_tabrecovery = tabrecovery.loc[tabrecovery["reliability"] == 0]
    print("shape true=", true_tabrecovery.shape)
    print("shape false=", fake_tabrecovery.shape)

    # split 80/20 true news into train/test and again for fake news
    train_true, test_true = train_test_split(true_tabrecovery, test_size=0.2)
    train_fake, test_fake = train_test_split(fake_tabrecovery, test_size=0.2)
    train_true["type"] = "training"
    test_true["type"] = "test"
    train_fake["type"] = "training"
    test_fake["type"] = "test"
    #concat and shuffle
    true_tabrecovery = pd.concat([train_true, test_true])
    true_tabrecovery = shuffle(true_tabrecovery)
    fake_tabrecovery = pd.concat([train_fake, test_fake])
    fake_tabrecovery = shuffle(fake_tabrecovery)
    # concatenate false and true news dataframes into single dataframe and shuffle again
    tabrecovery = pd.concat([true_tabrecovery, fake_tabrecovery], ignore_index=True)
    tabrecovery = shuffle(tabrecovery)
    #save data
    tabrecovery.to_csv(savepath + "/{}.csv".format(args.dataset), index=False)
    # tabrecovery.to_csv(savepath + "/Recov.csv".format(args.dataset), index=False)
    print("shape=", tabrecovery.shape)
    print("*****************************************************************************")
    print(" Data saved")
    print("*****************************************************************************")

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("-d", "--dataset", help="Dataset name", default="")
    args = parser.parse_args()
    dataset = args.dataset


    if dataset == "ReCOVery":
        datapath = os.getcwd() + '/data/ReCOVery/dataset/recovery-news-data.csv'
        savepath = os.getcwd() + "/data/ReCOVery/dataset/"
        prepare_recovery(datapath,savepath,args)