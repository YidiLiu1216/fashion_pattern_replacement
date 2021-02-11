pattern_replacement
===================

Table of Contents
-------------
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Preparedata](#Preparedata)
3. [Train](#train)
4. [Tests](#tests)
5. [Datasets](#datasets)

Introduction
-------------
The pattern_replacement program replace input cloth with given pattern based on cloth trend. This process can remove original pattern on cloth.

![input](/data/demo_image/demo.png)

Installation
------------- 
**Requirements shortlists:**
- Pattern replacement program
- Prepare data

**Run demo on given data requires:**

- NVIDIA GPU, Python3
- Tensorflow-gpu
- various standard Python packages in requirement.txt
- Pillow

Notes:
- The program was tested on tensorflow-gpu 2.3.1 with cuda 10.1 and cudnn 7.6
- The program requires your opencv-contrib-python and opencv-python in the same version 

**Prepare your own data:**

To run you own data on this demo, you will need to prepare the [Segmentation](https://github.com/PeikeLi/Self-Correction-Human-Parsing) result of image, and run [Densepose](https://github.com/facebookresearch/DensePose) to get the human body trend information.

The main requirement of these programs:
- [Caffe2,COCOAPI](https://github.com/facebookresearch/DensePose/blob/master/INSTALL.md)
- PyTorch >= 1.4.0

The details requirement of [Segmentation](https://github.com/PeikeLi/Self-Correction-Human-Parsing) and [Densepose](https://github.com/facebookresearch/DensePose) can also be found in the link.

Preparedata
-------------
1. Put input image under data/input/
2. Run segementation program and put result under data/seg/
3. Run
```
python preparedata.py
```
The result will under data/mask/

4. Run densepose and put result under data/IUV/

**After done the requirement prepare work:**

Clone repository:
```
git clone https://github.com/aircat1216/fashion_pattern_replacement your_path
```
Install Python dependencies:
```
pip install -r your_path/requirements.txt
```
Train
-------------
Tests
-------------
If you want to run given demo, just run:
```
python predict.py
```
And you will find gray_scale result in data/output_gray/ and finan result in data/output_final.

To use your own data:
1. Put the input images under data/input/
2. [preparedata](#Preparedata) and put mask under data/mask, IUV images under data/images
3. run
```
python predict.py
```
If you want to select your own model and pattern, run:
```
python predict.py your_model_weight_file_name.h5 your_pattern_path
```
See [Preparedata](#Preparedata) to prepare your own data

Datasets
-------------
