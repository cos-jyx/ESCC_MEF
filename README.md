# ESCC_MEF
This is the implementation of the submitted paper “Development and Validation of a Prediction Model for Treatment-related Malignant Fistulas in Advanced Esophageal Squamous Cell Carcinoma”.<br>
Here is the main framework of the model
![framework](https://github.com/cos-jyx/ESCC_MEF/blob/main/picture/framework.png)

# Data
To facilitate model training, the data used should be cropped out of the oesophageal cancer region after preprocessing with a size of (10,32,32) for model training and inference.
# Requirements
This code has been tested On Ubuntu 20.04.<br>
The required python package versions are shown below:<br>
Python =3.8.6<br>
numpy =1.23.1<br>
tensorflow =2.12.0<br>
six =1.16.0<br>
sklearn =1.1.1<br>
matplotlib =3.7.3<br>
pandas =1.4.3<br>
tqdm =4.64.0<br>
SimpleITK =2.2.0<br>
# Installation guide
First, install Miniconda on your machine (download the distribution that comes with python3).<br>
```
      wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py38_4.8.3-Linux-x86_64.sh --no-check-certificate
      bash Miniconda3-py38_4.8.3-Linux-x86_64.sh
      conda -V
```
After setting up Miniconda, instal SimpleITK
```
      conda install simpleitk
```
Then, create a conda environment
```
      conda create -n ESCC_MEF python=3.8.6
```
Activate the environment
```
      conda activate ESCC_MEF
```
Finally, install the python package as described above.
# How to run
1. Cropping of oesophageal cancer tumour region image using Rertangle.py file<br>
2. Run train.py to train the model<br>
3. Run test.py to test the model<br>
# Model performance
If it runs properly, you will get the following results
![auc](https://github.com/cos-jyx/ESCC_MEF/blob/main/picture/auc.png)
