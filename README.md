# Overview

This project is an extention of Itay Hubara's PyTorch implementation of binary neural networks. The original code can be found here:  
https://github.com/eladhoffer/convNet.pytorch
The objective of this project is to extend the work done by Itay Hubara and his fellow colleagues in the field of BNNs.  

## Team Members

1. **George Kotobuki**: Electrical and Computer Engineering Department, College of Engineering, California State Polytechnic University, Pomona. 
    
## Supervising Professor 

**Mohamed El-Hadedy:** Assistant Professor, Electrical and Computer Engineering department, College of Engineering, California State Polytechnic University, Pomona.



### What are binary neural networks(BNNs)?

BNNs are deep quantized neural networks that aim to reduce the computational resources required by deep neural networks by using binary weights and activations.


### Setup
1. Install Anaconda by following the instructions on the official page:  
    - https://docs.anaconda.com/anaconda/install/
2. Create and configure the virtual environment:
    - First create the virtual environment:
      - ``` conda create --name myenv ```
      - where 'myenv' corresponds to the name of the virtual environment and can be substituted for some other name.
    - Activate the virtual environment with the following command:
      - ``` conda activate myenv ```
    - First install the supported version of Python.
      - ``` conda install python=3.7```
    - Install the packages and libraries necessary:
      - ```conda install pytorch torchvision cudatoolkit=10.2 -c pytorch```
      - ```conda install -y pandas```
      - ```conda install -y bokeh```
      - ```conda install -y scipy```

### Usage
Run the following command to see the argument options.   
  ```python main_binary.py --help```  
  
Example: Run the Binary ResNet18 model on the CIFAR-10 dataset and save the data to **results/resnet18_binary**  
  ```python main_binary.py --model resnet_binary --save resnet18_binary --dataset cifar10```   

### Current project state and future plans
1. Currently the models are not true binary models. The models use binary weights and activations in the forward pass, but store the floating-point weights, which are used during backpropagation. It is necessary to implement a straight through estimator in order to perform backpropogation without needing to store the full precision weights.
2. Implement support for binarized versions of other layers, such as RSTM, RNN layers.
3. Implement function to convert FP model into its binary equivalent.
