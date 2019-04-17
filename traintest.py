import torch
import cv2
import os
import numpy as np
from os import listdir
from os.path import isfile, isdir, join
import models
import loss
import loaddata




def main(images_path="hw2_train_val/val1500/images",targets_path="hw2_train_val/val1500/labelTxt_hbb"):

    theshold=0.1



    testloss=0.
    count=0
    imgnames,imgs,targets=loaddata.read_val_data(images_path,targets_path)

    net=models.Yolov1_vgg16bn(pretrained=True)
    #
    net.cuda()
    print(type(net))
    net.load_state_dict(torch.load("saved_model10.pth"))

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

    #for



    print(testloss/count)




if __name__ == '__main__':
    main()
