##Introduction
This repository provides the code for the methods and experiments presented in our paper 'ECPS: Cross Pseudo Supervision Based on Ensemble Learning for Semi-Supervised Remote Sensing Change Detection'. (TGRS2024)
![image](https://github.com/TangXu-Group/ECPS/assets/74549002/e054afe5-0c60-4840-a560-c9f833f0b4bf)
If you have any questions, you can send me an email. My mail address is yqunyang@163.com

##Dataset loading
CDD, LEVIR-CD and SYSU-CD.
For each dataset, please prepare four txt files:
wlabel.txt for listing labeled training data;
olabel.txt for listing unlabeled training data;
val.txt for listing validation data; and
test.txt for listing test data.
Here, each txt file list all items with the following form:
'''
<the path of time1 image> <the path of time2 image> <the path of label>
<the path of time1 image> <the path of time2 image> <the path of label>
...
<the path of time1 image> <the path of time2 image> <the path of label>
'''
For example:
![image](https://github.com/TangXu-Group/ECPS/assets/74549002/9c9901f9-3364-4e0f-aed8-b056bba19852)


