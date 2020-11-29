# Cross-lingual structured flow model for zero-shot syntactic transfer 

This is PyTorch implementation of the [paper](https://arxiv.org/abs/1906.02656):
```
Cross-Lingual Syntactic Transfer through Unsupervised Adaptation of Invertible Projections
Junxian He, Zhisong Zhang, Taylor Berg-Kiripatrick, Graham Neubig
ACL 2019
```

The structured flow model is a generative model that can be trained in an supervised fashion on labeled data in another language, but also perform unsupervised training to directly maximize likelihood of the target language. In this way, it is able to transfer shared linguistic knowledge from the source language as well as learning language-specific knowledge on the unlabeled target language.

Please concact junxianh@cs.cmu.edu if you have any questions.

## Requirements

- Python >= 3.6
- PyTorch >= 0.4 (the code is tested up to version 1.7.0)

Additional requirements can be installed via:
```bash
pip install -r requirements.txt
```

## Data
Download the Universal Dependencies 2.2 [here](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-2837) (`ud-treebanks-v2.2.tgz`), put file `ud-treebanks-v2.2.tgz` into the top-level directory of this repo, and run:
```
$ tar -xvzf ud-treebanks-v2.2.tgz
$ rm ud-treebanks-v2.2.tgz
```

## Prepare Embeddings
### fastText
The fastText embeddings can be downloaded in the Facebook fastText [repo](https://github.com/facebookresearch/fastText/blob/master/docs/pretrained-vectors.md) (Note that there are different versions of pretrained fastText embeddings in the fastText repo, but the embeddings must be downloaded from the given link since the alignment matrices (from [here](https://github.com/Babylonpartners/fastText_multilingual)) we used are learned on this specific version of fastText embeddings). Download the fastText model `bin` file and put it into the `fastText_data` folder.

Take English language as an example to download and preprocess the fastText embeddings:
```
$ cd fastText_data
$ wget https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.zip
$ unzip wiki.en.zip
$ cd ..

# create a subset of embedding dict for faster embedding loading and memory efficiency
$ python scripts/compress_vec_dict.py --lang en

$ rm wiki.en.vec
$ rm wiki.en.zip
```

The argument for `--lang` is the short code of the language, the list of short codes and corresponding languages is in `statistics/lang_list.txt`.

### multilingual BERT (mBERT)
```
$ CUDA_VISIBLE_DEVICES=xx python scripts/create_cwr.py --lang [language code]
```
This command pre-computes the BERT contexualized word representations using [pytorch-pretrained-BERT](https://github.com/huggingface/pytorch-pretrained-BERT) for each sentence in the corresponding treebank. These embeddings are saved in `bert-base-multilingual-cased` (will be created automatically) as `hdf5` files. This command would download the pretrained multilingual BERT model and cache it when it is executed for the first time. 

## POS Tagging
Several training scripts are provided (note that supervised training scripts must be run first before running unsupervised training scripts):
```
# supervised training on English with fastText
# [gpu_id] is an integer number
$ ./scripts/run_supervised_tagger.sh [gpu_id]

# unsupervised training on other languages with fastText
$ ./scripts/run_unsupervised_tagger.sh [gpu_id] [language code]




# supervised training on English with mBERT
$ ./scripts/run_supervised_bert_tagger.sh [gpu_id]

# unsupervised training on other languages with mBERT
$ ./scripts/run_unsupervised_bert_tagger.sh [gpu_id] [language code]
``` 
Trained models and logs are saved in `outputs/tagging`. 

## Dependency Parsing
Several training scripts are provided (note that supervised training scripts must be run first before running unsupervised training scripts):
```
# supervised training on English with fastText
# [gpu_id] is an integer number
$ ./scripts/run_supervised_parser.sh [gpu_id]

# unsupervised training on distant languages with fastText
$ ./scripts/run_unsupervised_parser_distant.sh [gpu_id] [language code]

# unsupervised training on nearby languages with fastText
$ ./scripts/run_unsupervised_parser_nearby.sh [gpu_id] [language code]




# supervised training on English with mBERT
# [gpu_id] is an integer number
$ ./scripts/run_supervised_bert_parser.sh [gpu_id]

# unsupervised training on distant languages with mBERT
$ ./scripts/run_unsupervised_bert_parser_distant.sh [gpu_id] [language code]

# unsupervised training on nearby languages with mBERT
$ ./scripts/run_unsupervised_bert_parser_nearby.sh [gpu_id] [language code]
```
Trained models and logs are saved in `outputs/parsing`.

## Acknowledgement
This project would not be possible without the [URIEL](http://www.cs.cmu.edu/~dmortens/uriel.html) linguistic database, pre-computed [fastText alignment matrix](https://github.com/Babylonpartners/fastText_multilingual), Google's pretrained [multilingual BERT model](https://github.com/google-research/bert), and the pyTorch interface of pretrained BERT models [pytorch-pretrained-BERT](https://github.com/huggingface/pytorch-pretrained-BERT).

## Reference
```
@inproceedings{he19acl,
    title = {Cross-Lingual Syntactic Transfer through Unsupervised Adaptation of Invertible Projections},
    author = {Junxian He and Zhisong Zhang and Taylor Berg-Kirkpatrick and Graham Neubig},
    booktitle = {Proceedings of ACL},
    year = {2019}
}

```

