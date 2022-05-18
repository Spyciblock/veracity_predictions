
## Onclusive Machine Learning Challenge (Anthony T.)
This is the code of the system proposed in my previous paper on a custom project related to Fake
news detection, this is based on the paper cited below.

I experiment this method because I thought the affective information in text is still important in this
topic specific dataset PubHealths.
> The experiments are in the notebook Fakeflow.ipynb

#### FakeFlow: Fake News Detection by Modeling the Flow of Affective Information

REQUIREMENTS:
- gensim==3.8.0
- joblib==0.14.1
- Keras==2.2.4
- Keras-Preprocessing==1.1.1
- keras-self-attention==0.35.0
- numpy==1.16.0
- pandas==0.24.2
- nltk==3.4.5
- scikit-learn==0.20.2
- tensorflow-gpu==1.14.0
- tqdm==4.32.1
- hyperopt==0.1.1
- tensorflow==2.4.0


1) Place your data in the folder `./data/DATASET_NAME`
like `/data/ReCOVeryt/..txt` if you already have.
2) Install GoogleNEws, and put in fake_flow directory then unzip:
 from
- `https://code.google.com/archive/p/word2vec/`

- `gzip -d GoogleNews-vectors-negative300.bin.gz` 

3) If we use the additional Recovery dataset, collect and prepare the dataset:
Initialize train/test data splits for Recovery  with:

Parameter: 
`-d`: dataset name (i.e. ReCOVery, Celebrity).
- Run: `python splits.py -d ReCOVery`
-----------------------------
Parameters:
`-d`: dataset name (i.e. ReCOVery, PubHealth ).

`-otherd`: second dataset name (i.e.  ReCOVery ).

`-sn`: number of segments.

`-epochs`: number of epochs (default 50).

`-batch_size`: batch_size (default 64).

`-s`: to search for params; enter a number larger than 0 to search for N different combination of parameters (e.g. 150).

`-m`: mode (train or test); if you want to load a pretrained model.

`-c`: mode (True/False by default) to combine 2 datasets in training part and test on PubHealth

`-newproc`: mode (True/False by default) to use my custom model

You have to change datasets name( -d and -otherd).

To use the custom model, use previous examples code, and add 
`-newproc True`

Some examples:Use only the dataset PubHealth:
> python fake_flow.py -d PubHealth -sn 10

To load saved model after training=testing:
> python fake_flow.py -d PubHealth -sn 10 -m test

To search for best params:
> python fake_flow.py -d PubHealth -s 80

Optional:

To train with both PubHealth and ReCOVery datasets and test on PubHealth:
> python fake_flow.py -d PubHealth -otherd ReCOVery -sn 10 -m train -c True

To search for best params with training on PubHealth and ReCOVery datasets and test on PubHealth:
> python fake_flow.py -d PubHealth -otherd ReCOVery -sn 10 -m train -c True -s 60 

-------------------------------------------------------------------------------------------------

### For running models using Huggingface library with simpletransformers

requirements: 
- transformers
- datasets
- simpletransformers
- numpy
- pandas
- seaborn
- nltk

I experiment with ROBERTA and ALBERT models. 
> The experiments are in the notebook Onclusive_ML_Challenge.ipynb

Parameters:

`-path`: define the path to save the output (optional)

`-model`: model to use (string), "roberta","bert","albert"...

`-modelversion`: select a version of the model (string),"roberta-base","bert-base-uncased","albert-base-v2"...

`-path`: Path to save results (optional)

`-outputdir`: Path of directory to save checkpoints

`-use_cuda`: Train with GPU (default True)

Run the model :
- `python trainer.py -model "roberta" -modelversion "roberta-base" -path "outputs/"`


Citation:

    @inproceedings{ghanem2021fakeflow,
      title={{FakeFlow: Fake News Detection by Modeling the Flow of Affective Information}},
      author={Ghanem, Bilal and Ponzetto, Simone Paolo and Rosso, Paolo and Rangel, Francisco},
      booktitle={Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics},
      year={2021}
    }
