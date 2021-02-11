pattern_replacement
===================

Table of Contents
-------------
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Preparedata](#preparedata)
3. [Train](#train)
4. [Tests](#tests)
5. [Datasets](#datasets)

Introduction
-------------
The pattern_replacement program replace input cloth with given pattern based on cloth trend. This process can remove original pattern on cloth.

![input](/data/demo_image/demo.png)

Installation
------------- 
Requirements shortlists:
Pattern replacement program
Prepare data

Run demo on given data:

- NVIDIA GPU, Python3
- Tensorflow-gpu
- various standard Python packages

Notes:
- The program was tested on tensorflow-gpu 2.3.1 with cuda 10.1 and cudnn
- The program requires your opencv-contrib-python and opencv-python in the same version 

Prepare Your own data:

To run you own data on this demo, you will need to prepare the [Segmentation](https://github.com/PeikeLi/Self-Correction-Human-Parsing) result of image, and run [Densepose](https://github.com/facebookresearch/DensePose) to get the human body trend information.

The requirement of these programs:

Preparedata
-------------
Train
-------------
Tests
-------------
Datasets
-------------
