import torch
import cv2
import os
import numpy as np
from os import listdir
from os.path import isfile, isdir, join
import models
import loss
import loaddata

category=(
"plane",
"ship",
"storage-tank",
"baseball-diamond",
"tennis-court",
"basketball-court",
"ground-track-field",
"harbor",
"bridge",
"small-vehicle",
"large-vehicle",
"helicopter",
"roundabout",
"soccer-ball-field",
"swimming-pool",
"container-crane")

def _iou(boxes1 , boxes2 ):

    #iou[n,m]=boxes[n]和boxes2[m]的iou x,y,w,h
    """
    xy1_1=boxes_xywh1[:,:2]-(boxes_xywh1[:,2:])*0.5*7
    xy2_1=boxes_xywh1[:,:2]+(boxes_xywh1[:,2:])*0.5*7
    boxes1=torch.cat((xy1_1,xy2_1),1)

    xy1_2=boxes_xywh2[:,:2]-(boxes_xywh2[:,2:])*0.5*7
    xy2_2=boxes_xywh2[:,:2]+(boxes_xywh2[:,2:])*0.5*7
    boxes2=torch.cat((xy1_2,xy2_2),1)
    """
    n1=boxes1.size(0)
    n2=boxes2.size(0)
    x1y1max=torch.max(boxes1[:,:2].unsqueeze(1).expand(n1,n2,2),boxes2[:,:2].unsqueeze(0).expand(n1,n2,2),)
    x2y2min=torch.min(boxes1[:,2:].unsqueeze(1).expand(n1,n2,2),boxes2[:,2:].unsqueeze(0).expand(n1,n2,2),)

    inter_size= x2y2min - x1y1max
    inter_size[inter_size<0]=0.
    inter_area=inter_size[:,:,0]*inter_size[:,:,1]

    area1 = ((boxes1[:,2]-boxes1[:,0])*(boxes1[:,3]-boxes1[:,1])).unsqueeze(1).expand(n1,n2)
    area2 = ((boxes2[:,2]-boxes2[:,0])*(boxes2[:,3]-boxes2[:,1])).unsqueeze(0).expand(n1,n2)
    iou = inter_area/(area1+area2-inter_area)

    return iou


def main(images_path="hw2_train_val/val1500/images",targets_path="hw2_train_val/val1500/labelTxt_hbb"):

    theshold=0.1



    testloss=0.
    count=0
    imgnames,imgs,targets=loaddata.read_val_data(images_path,targets_path)

    net=models.Yolov1_vgg16bn(pretrained=True)
    #
    net.cuda()
    print(type(net))
    net.load_state_dict(torch.load("train_model66.pth"))

    for i in range(imgs.size(0)):
        img_3_448_448=torch.stack((imgs[i][:,:,0],imgs[i][:,:,1],imgs[i][:,:,2]))
        img_3_448_448=img_3_448_448.cuda()
        img_3_448_448=img_3_448_448.unsqueeze(0)
        img_3_448_448=img_3_448_448.float()


        output=net(img_3_448_448)
        t=targets[i].unsqueeze(0)
        t=t.cuda()


        testloss=testloss+float(loss.trainloss(output,t,5,0.5))
        count+=1
        """
        for j in range(16):
            looked_data=output[0][:,:,:4]
        """



    print(testloss/count)




if __name__ == '__main__':
    main()
