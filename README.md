# binary_networks

The purpose of this repo is to get insights into binary networks. 

Namely, it contains an implementation of [BinaryConnect](https://proceedings.neurips.cc/paper/2015/file/3e15cc11f979ed25912dff5b0669f2cd-Paper.pdf) where the weights of the neural network are binarized : 

```
Binaryconnect: Training deep neural networks with binary weights during propagations. 
Courbariaux, Matthieu and Bengio, Yoshua and David, Jean-Pierre. NIPS 2015
```
and a [second paper](https://arxiv.org/pdf/1602.02830.pdf) where both weights and activations are binarized.
```
Binarized neural networks. 
Hubara, Itay and Courbariaux, Matthieu and Soudry, Daniel and El-Yaniv, Ran and Bengio, Yoshua. NIPS 2016.
```

These models are implemented on a very simple setup to better highlight the specifity of the binarisation. We used fully connected layer architectures with 100 hidden units which are learned on cyfar dataset. 


## requirements
```
pytorch >= 1.9
```

## Usage


Linear model with Binary connect
```
python run_exp_BinaryConnect_Lineaire.py --data_folder DATA_FOLDER --config_file config_gradient_1_layer.json
```

One hidden layer architecture with binary connect
```
python run_exp_BinaryConnect_Perceptron.py --data_folder DATA_FOLDER --config_file config_gradient_100_layer.json
```

One hidden layer architecture with binary network
```
python run_exp_BinaryNetwork_Perceptron.py  --data_folder DATA_FOLDER --config_file config_gradient_100_layer.json
```




