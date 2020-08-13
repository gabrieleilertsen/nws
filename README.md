# **Dissecting the weight space of neural networks**

## General
We have trained a collection of 16,000 deep convolutional neural networks, for the purpose of performing a dissection of the weight space of neural nets. This repository provides scripts for training additional networks, as well as training meta-classifiers. A meta-classifier takes the sampled weight space as training data input and the objective is to classify which hyper-parameters were used for training a specific weight sample. For example, given only the weights of a network, can we predict which dataset was used in training the model? The purpose of a meta-classifier is to probe for information in neural network weights, in order to learn about how optimization locally shapes the weights of a network.

For information on the study of the weight space, we refer to the [paper](https://arxiv.org/abs/2002.05688), which will be presented at the [European Conference on Artificial Intelligence](http://ecai2020.eu/) (ECAI 2020). For now, the paper can be found on arXiv, and if you use the code or dataset please cite according to:

```
@article{EJRUY20,
  author       = "Eilertsen, Gabriel and J\"onsson, Daniel and Ropinski, Timo and Unger, Jonas and Ynnerman, Anders",
  title        = "Classifying the classifier: dissecting the weight space of neural networks",
  journal=     = "arXiv preprint arXiv:2002.05688",
  year         = "2020"
}
```

## Dataset
The entire dataset of neural network weights is available 
[here](https://liuonline-my.sharepoint.com/:f:/g/personal/gabei62_liu_se/ErHWT-psvCNPr0yyQmybqfUBSMIjyO7LNMGTdUBRcIEj1Q). The dataset consists of 16,000 CNNs, captured at 20 different steps during training, from initialization to converged model. This means that the dataset in total is comprised of 320,000 CNN weight snapshots.

The dataset contains two different subsets: 
* **nws_fixed**: 3,000 trained nets with fixed architecture, and randomly selected dataset and hyper-parameters (see the paper for details).
* **nws_main**: 13,000 trained nets with randomly selected architecture, dataset and hyper-parameters.

Each of these has been split into a training and test set, by means of the provided CSV files. Also, the CSV files contain information about the architecture and hyper-parameters of each trained net.

Providing the path to a certain CSV file, [`util.import_weights()`](https://github.com/gabrieleilertsen/nws/blob/9c131051ad7c391e502b2a84bd6f9dd4f9daa55a/util.py#L81) can be used to read a dataset with annotations for a selected hyper-parameter.

## Usage
#### Sampling the weight space
**`classifier.py`** is used to sample/train a network with randomly selected hyper-parameters, e.g.:

`python classifier.py --data_nmn=mnist --log_dir=OUTPUT_PATH --prints=20 --fixed=0`

`prints` selects how many times during a training weights should be exported. `fixed` can be used to fix the architecture, i.e. so architectural parameters are the same in each training.

#### Training on the weight space dataset
**`meta_classifier.py`** can train and evaluate a meta-classifier, i.e. classification of raw weights given some chosen hyper-parameter. For example, this could use the **nws_main** dataset to learn how to classify which dataset a weight sample was trained on:

`python meta_classifier.py --data_train=nws_main_train.csv --data_test=nws_main_test.csv --prop=0 --K=11,20 --slice_length=5000`

`prop` selects which hyper-parameter to classify. `K` specifies which snapshots of a network to use in training, in this case snapshots 11-20 (each model of the dataset has 20 weight snapshots, from initialization to converged model). `slice_length` selects how large subset of weights to use in training, i.e. it specifies the slice of weights for *local* meta-classification.

## License

Copyright (c) 2020, Gabriel Eilertsen.
All rights reserved.

The code is distributed under a BSD license. See `LICENSE` for information.
