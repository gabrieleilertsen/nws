# **Dissecting the weight space of neural networks**

## General
We have trained a collection of 16,000 deep convolutional neural networks, for the purpose of performing a dissection of the weight space of neural nets. This repository provides scripts for training additional networks, as well as training meta-classifiers. A meta-classifier takes the sampled weight space as training data input and the objective is to classify which hyper-parameters were used for training a specific weight sample. For example, given only the weights of a network, can we predict which dataset was used in training the model? The purpose of a meta-classifier is to probe for information in neural network weights, in order to learn about how optimization locally shapes the weights of a network.

For information on the study of the weight space, we refer to the [paper](https://arxiv.org/abs/xxxx.yyyy), which will be presented at the [European Conference on Artificial Intelligence](http://ecai2020.eu/) (ECAI 2020). For now, the paper can be found on arXiv, and if you use the code or dataset please cite according to:

```
@article{EJRUY20,
  author       = "Eilertsen, Gabriel and J\"onsson, Daniel and Ropinski, Timo and Unger, Jonas and Ynnerman, Anders",
  title        = "Classifying the classifier: dissecting the weight space of neural networks",
  journal=     = "arXiv preprint arXiv:xxxx.yyyy",
  year         = "2020"
}
```

## Dataset
The entire dataset of neural network weights is available 
[here](https://liuonline-my.sharepoint.com/:f:/g/personal/gabei62_liu_se/ErHWT-psvCNPr0yyQmybqfUBSMIjyO7LNMGTdUBRcIEj1Q). The dataset consists of 16,000 CNNs, captured at 20 different steps during training, from initialization to converged model. This means that the dataset in total is comprised of 320,000 CNN weight snapshots.

## Usage
Information on how to use the weight dissection methods will be provided soon...

## License

Copyright (c) 2020, Gabriel Eilertsen.
All rights reserved.

The code is distributed under a BSD license. See `LICENSE` for information.
