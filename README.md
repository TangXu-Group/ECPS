#Introduction
=
This repository provides the code for the methods and experiments presented in our paper 'ECPS: Cross Pseudo Supervision Based on Ensemble Learning for Semi-Supervised Remote Sensing Change Detection'. (TGRS2024)
![image](https://github.com/TangXu-Group/ECPS/assets/74549002/e054afe5-0c60-4840-a560-c9f833f0b4bf)
If you have any questions, you can send me an email. My mail address is yqunyang@163.com

#Dataset loading
=
For the datasets CDD, LEVIR-CD, SYSU-CD and OSCD, it is required to prepare four text files for each dataset. The files are as follows:
`wlabel.txt` for listing the labeled training data;
`olabel.txt` for listing the unlabeled training data;
`val.txt` for listing the validation data; and
`test.txt` for listing the test data.
Each text file should list all items in the following format:
```
<the path of time1 image> <the path of time2 image> <the path of label>
<the path of time1 image> <the path of time2 image> <the path of label>
...
<the path of time1 image> <the path of time2 image> <the path of label>
```
For example:
![image](https://github.com/TangXu-Group/ECPS/assets/74549002/9c9901f9-3364-4e0f-aed8-b056bba19852)

#Training
##Prepare
=
The based dependencies of runing codes:
```
torchvision: 0.9.0
torch: 1.8.0
python: 3.8.17
```
OR
Using `requirements.txt`

##Start
=
```
python train.py -d CDD -t 0.05 -s 2
```
Here, `-d` denotes the name of dataset ("CDD", "LEVIR", "SYSU" and "OSCD"), `-t` represents the training ratio (0.05, 0.1, 0.2 and 0.4), and `-s` is the number of students.


