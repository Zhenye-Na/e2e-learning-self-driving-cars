# Self-driving Vehicles Simulation using Machine Learning

<p align="center">
  <img src="./img/driverless-car.png" width="10%">
</p>

> Please install [Open in Colab](https://chrome.google.com/webstore/detail/open-in-colab/iogfkhleblhcpcekbiedikdehleodpjo) extension in Google Chrome in order to open a Github-hosted notebook in Google Colab with one-click.



## Table of Contents

- [Introduction](#introduction)
- [Prerequisite](#prerequisite)
- [Usage](#usage)
- [Demo videos](#demo-videos)
- [Project guide](#project-guide)
    - [1. Prepare training data](#1-prepare-training-data)
    - [2. Project code structure](#2-project-code-structure)
    - [3. Training neural network](#3-training-neural-network)
    - [4. Model architecture](#4-model-architecture)
    - [5. Test model performance](#5-test-model-performance)
- [FAQ](#faq)
- [References](#references)



## Introduction

Self-driving vehicles is the most hottest and interesting topics of research and business nowadays. More and more giant companies have jumped into this area. In this project, I will show you the power of Deep Neural Network in this field.



## Prerequisite

We will use Python as the primary programming language and [PyTorch](https://pytorch.org/) as the Deep Learning framework. Other resources / software / library could be found as follows.


[1] Self-driving car simulator developed by [Udacity](https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013) with Unity. Download [here](https://github.com/udacity/self-driving-car-sim)  
[2] Install [PyTorch environment](https://pytorch.org/get-started/locally/) in your local machine.  
[3] Register account in [Google Colab](https://colab.research.google.com/) (if you do not have GPU and would love to utilize the power of GPU, please try this and be sure to enable `GPU` as accelerator)  
[4] Nvidia research paper: [End to End Learning for Self-Driving Cars](https://arxiv.org/pdf/1604.07316v1.pdf)  
[5] Training data for [track 1]() and [track 2]() (Optional).  


## Usage

1. Download this repo.


    ```sh
    git clone 
    
    cd 
    ```

2. Type the following snippet in your terminal and open the "Self-driving car simulator"

    ```sh
    python3 drive.py model.h5 
    ```

3. Click **`Allow`** to accept incoming network connections for python scripts.



### Demo videos


| [![Watch the video](./img/bilibili1.jpg)](https://www.bilibili.com/video/av47638211/) | [![Watch the video](./img/bilibili1.jpg)](https://www.bilibili.com/video/av47638211/) |
|:--------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------:|


## Project guide

### 1. Prepare training data

Once you install the self-driving car simulator, you will find there are 2 different tracks in the simulator. The second (right-hand side) one is much harder to train. Please choose the terrain you like first and make sure that you select **`TRAINING MODE`**. 

<p align="center">
  <img src="./img/main-menu.png" width="80%">
</p><br>

Click **`RECORD`** button on the right corner and select a directory as the folder to save your training image and driving log information.

<p align="center">
  <img src="./img/recording.png" width="80%">
</p><br>


<p align="center">
  <img src="./img/select-folder.png" width="80%">
</p><br>


Click **`RECORD`** again and move your car smoothly and carefully.


After you have complete recording your move, the training data will be stored in the folder you selected. Here I suggest you record at least 3 laps of the race. The first lap of race, please try best to stay at the center of the road, the rest could be either on the left hand side and right hand side of the road separately.


- `/IMG/` - contains training images from three directions of the car - center image, left image and right image.
- `driving_log.csv` - save the image name information and corresponding information like steer angle, current speed at that time.


### 2. Project code structure



pass


### 3. Training neural network

<p align="center">
  <img src="./img/training.png" width="80%">
</p><br>

pass



### 4. Model architecture

<p align="center">
  <img src="./img/model-achitecture.png" width="80%">
</p><br>

pass



### 5. Test model performance

After training process, use the saved model and `drive.py` file to test your model performance in the simulator. Remeber to select **`AUTONOMOUS MODE`**.

```sh
python3 drive.py model.h5 
```



## FAQ


### `AttributeError: Can't get attribute '_rebuild_parameter'`


This error means that you are trying to load a newer version of model checkpoint in an older version of PyTorch.

Check your PyTorch version:

```python
import pytorch
print(torch.__version__)
```


For Google Colab users, please try to first get your PyTorch version in your local machine and install the same version on Colab virtual machine using the following snippets, PyTorch `0.4.1`, for example.

```python
# http://pytorch.org/
from os.path import exists
from wheel.pep425tags import get_abbr_impl, get_impl_ver, get_abi_tag
platform = '{}{}-{}'.format(get_abbr_impl(), get_impl_ver(), get_abi_tag())
cuda_output = !ldconfig -p|grep cudart.so|sed -e 's/.*\.\([0-9]*\)\.\([0-9]*\)$/cu\1\2/'
accelerator = cuda_output[0] if exists('/dev/nvidia0') else 'cpu'

!pip install -q http://download.pytorch.org/whl/{accelerator}/torch-0.4.1-{platform}-linux_x86_64.whl torchvision
```

If it is not working, trying to use the following codes instead to install PyTorch `0.4.1`, for example.


```python
from os import path
from wheel.pep425tags import get_abbr_impl, get_impl_ver, get_abi_tag
platform = '{}{}-{}'.format(get_abbr_impl(), get_impl_ver(), get_abi_tag())

accelerator = 'cu80' if path.exists('/opt/bin/nvidia-smi') else 'cpu'

!pip install -q http://download.pytorch.org/whl/{accelerator}/torch-0.4.1-{platform}-linux_x86_64.whl torchvision
import torch
print(torch.__version__)
print(torch.cuda.is_available())
```

```
Output:

0.4.1
True
```


## References

[1] Nvidia research, [End to End Learning for Self-Driving Cars](https://arxiv.org/pdf/1604.07316v1.pdf)  
[2] Self-driving car simulator developed by [Udacity](https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013) with Unity  



* * *

<div>Icons made by <a href="https://www.flaticon.com/authors/eucalyp" title="Eucalyp">Eucalyp</a> from <a href="https://www.flaticon.com/" 			    title="Flaticon">www.flaticon.com</a> is licensed by <a href="http://creativecommons.org/licenses/by/3.0/" 			    title="Creative Commons BY 3.0" target="_blank">CC 3.0 BY</a></div>