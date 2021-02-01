# TAG
-----------------------------------------
Dataset:

You could download the dataset from here:

https://www.dropbox.com/sh/bpcpnxfhvpfuob3/AADPF_ExSgToZZTMrD8RcYnSa?dl=0

The dataset should be be put in /Data/Raw_Data

train_set.json

test_set.json

train_set_rand.json

test_set_rand.json

The original word vector file is too large, so a sub-dict of all words that show up in the train set and the test set are provided in /Embedding_Dict folder

-------------------------------------------
Code:

For Graph-Graph Matching model:

Modify cnofigs.py to change the number of layers, number of bases and all other parameters.

To run multiple set of parameters in a seire, set layers =[2,3,4....] and base = [1,2,3,4,.....]

Run Graph_Main.py for training

------------------------------------------------

For Graph-Squence Matching  model:

Main parameters are stored in AGMatch_Main.py

Run the .py file for training

------------------------------------------------

For Sequence Matching model:

Main parameters are stored in BERT_Main.py

Run the .py file for training

Modify /Model/matching.py to use CLS or Bi-LSTM for encoding


------------------------------------------------

For data preprocessing, in configs.py:

    IS_TAGEMB_GENERATED = True	means the embedding dict for POS,NER and SOURCE tag are already generated and stored in /Embedding_Dict
    
	----------------------
	
    IS_PREPROCESSED = True	means all data are preprocessed and stored in /Raw_Data; All tag embedding dict for POS,NER and SOURCE tag are already generated and stored in /Embedding_Dict
    
	----------------------
	
    IS_DICT_CUT = True	means  a sub_dict of all words in dataset has been generated and stored in /Embedding_Dict
    
------------------------------------------------

To change between overlapped dataset and non-overlapped dataset:
	
    USE_RAND_DATA = True means we will use overlapped dataset
