import torch
import cv2
import os
import numpy as np
from os import listdir
from os.path import isfile, isdir, join
category={
"plane":0,
"ship":1,
"storage-tank":2,
"baseball-diamond":3,
"tennis-court":4,
"basketball-court":5,
"ground-track-field":6,
"harbor":7,
"bridge":8,
"small-vehicle":9,
"large-vehicle":10,
"helicopter":11,
"roundabout":12,
"soccer-ball-field":13,
"swimming-pool":14,
"container-crane":15
}
"""
def iou_data(box1,box2):
    #xmin ymin xmax ymax

    x1y1max=torch.max(box1[:2],box2[:2])
    x2y2min=torch.min(box1[2:],box2[2:])
    print(x1y1max,x2y2min)
    inter_size= x2y2min - x1y1max
    inter_size[inter_size<0]=0.
    inter_area=inter_size[0]*inter_size[1]

    area1 = (box1[2]-box1[0])*(box1[3]-box1[1])
    area2 = (box2[2]-box2[0])*(box2[3]-box2[1])
    iou = inter_area/(area1+area2-inter_area)
    return iou
"""
def read_training_data():
    reading_num=15000




    imgs=[]
    for i in range(reading_num):
        img = cv2.imread("hw2_train_val/train15000/images/%05d.jpg"%(i),cv2.IMREAD_COLOR)
        img = cv2.resize(img,(448,448))
        imgs.append(torch.tensor(img))

    boxes=[]
    labels=[]
    answers=[]
    for i in range(reading_num):
        t = open("hw2_train_val/train15000/labelTxt_hbb/%05d.txt"%(i)).readlines()
        box=[]
        label=[]
        const =(448./512.)
        for j in range(len(t)):
            split_t=t[j].split( )
            box.append([float(split_t[0])*const , float(split_t[1])*const , float(split_t[4])*const , float(split_t[5])*const])#xmin ymin xmax ymax
            label.append(category[split_t[8]])#category
        boxes.append(torch.tensor(box))
        labels.append(torch.tensor(label))




    for i in range(reading_num):

        center=(boxes[i][:,2:]+boxes[i][:,:2])/2
        wh=(boxes[i][:,2:]-boxes[i][:,:2])
        grid_size=448./7
        flag = np.zeros((7,7),int)
        answer=torch.zeros(7,7,26)
        for j in range(len(labels[i])):
            n=int(center[j,0]/grid_size)#x
            m=int(center[j,1]/grid_size)#y

            gridnm=torch.tensor([n*grid_size,m*grid_size,(n+1)*grid_size,(m+1)*grid_size])

            if(not flag[n][m]):
                answer[n][m][0]=(center[j][0]-n*grid_size)/grid_size
                answer[n][m][1]=(center[j][1]-m*grid_size)/grid_size
                answer[n][m][2]=wh[j][0]/448
                answer[n][m][3]=wh[j][1]/448
                answer[n][m][4]=1
                answer[n][m][9]=1
                answer[n][m][10+labels[i][j]]=1
                flag[n][m]=1

        answers.append(answer)


    return torch.stack(imgs),torch.stack(answers)



def read_val_data(path1,path2):
    image_files = listdir(path1)
    targets_files = listdir(path2)



    imgs=[]

    for i in image_files:
        img = cv2.imread(path1+'/'+i,cv2.IMREAD_COLOR)

        img = cv2.resize(img,(448,448))
        imgs.append(torch.tensor(img))

    boxes=[]
    labels=[]
    answers=[]

    for i in targets_files:
        t = open(path2+'/'+i).readlines()

        box=[]
        label=[]
        const =(448./512.)
        for j in range(len(t)):
            split_t=t[j].split( )
            box.append([float(split_t[0])*const , float(split_t[1])*const , float(split_t[4])*const , float(split_t[5])*const])#xmin ymin xmax ymax
            label.append(category[split_t[8]])#category
        boxes.append(torch.tensor(box))
        labels.append(torch.tensor(label))





    for i in range(len(boxes)):

        center=(boxes[i][:,2:]+boxes[i][:,:2])/2
        wh=(boxes[i][:,2:]-boxes[i][:,:2])
        grid_size=448./7
        flag = np.zeros((7,7),int)
        answer=torch.zeros(7,7,26)
        for j in range(len(labels[i])):
            n=int(center[j,0]/grid_size)#x
            m=int(center[j,1]/grid_size)#y

            gridnm=torch.tensor([n*grid_size,m*grid_size,(n+1)*grid_size,(m+1)*grid_size])

            if(not flag[n][m]):
                answer[n][m][0]=(center[j][0]-n*grid_size)/grid_size
                answer[n][m][1]=(center[j][1]-m*grid_size)/grid_size
                answer[n][m][2]=wh[j][0]/448
                answer[n][m][3]=wh[j][1]/448
                answer[n][m][4]=1
                answer[n][m][9]=1
                answer[n][m][10+labels[i][j]]=1
                flag[n][m]=1

        answers.append(answer)


    return image_files,torch.stack(imgs),torch.stack(answers)
