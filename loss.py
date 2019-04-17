import torch

import numpy
def calculate_iou(boxes_xywh1 , boxes_xywh2 ):

    #iou[n,m]=boxes[n]和boxes2[m]的iou x,y,w,h
    xy1_1=boxes_xywh1[:,:2]-(boxes_xywh1[:,2:])*0.5*7
    xy2_1=boxes_xywh1[:,:2]+(boxes_xywh1[:,2:])*0.5*7
    boxes1=torch.cat((xy1_1,xy2_1),1)

    xy1_2=boxes_xywh2[:,:2]-(boxes_xywh2[:,2:])*0.5*7
    xy2_2=boxes_xywh2[:,:2]+(boxes_xywh2[:,2:])*0.5*7
    boxes2=torch.cat((xy1_2,xy2_2),1)

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

def trainloss(train_tensor,right_tensor,lamda_coord=5,lamda_noobj=0.5):
    n=train_tensor.size()[0]


    mask=(right_tensor[:,:,:,4]>0)
    mask=mask.unsqueeze(-1).expand_as(train_tensor)
    reversemask=(mask[:,:,:,:]==0)


    coord_train=train_tensor[mask].view(-1,26)
    coord_box_train=coord_train[:,:10].contiguous().view(-1,5)
    coord_class_train=coord_train[:,10:]

    coord_right=right_tensor[mask].view(-1,26)
    coord_box_right=coord_right[:,:10].contiguous().view(-1,5)
    coord_class_right=coord_right[:,10:]


    coord_loss=torch.cuda.FloatTensor([0.])
    obj_conf_loss=torch.cuda.FloatTensor([0.])
    noobj_c_loss=torch.cuda.FloatTensor([0.])
    for i in range(0,coord_box_train.size()[0],2):

        boxingrid1=coord_box_train[i:i+2,:4]
        boxingrid2=coord_box_right[i:i+2,:4]
        iou = calculate_iou(boxingrid1,boxingrid2)
        max_iou,label=iou.max(0)#label[i] means the box label[i] look at object[i]
        reverse_label=1-label


    #1
        for j in range(label.size()[0]):
            chosen=coord_box_train[label[j],:]
            notchosen=coord_box_train[reverse_label[j],:]

            train_xysqrtwh=torch.cat((chosen[:2],chosen[2:4]**0.5))
            right_xysqrtwh=torch.cat((coord_box_right[j][:2],coord_box_right[j][2:4]**0.5))
            #loss 1 2
            coord_loss= coord_loss + torch.nn.functional.mse_loss(train_xysqrtwh,right_xysqrtwh)
            obj_conf_loss = obj_conf_loss + (chosen[4]-coord_box_right[j][4]*max_iou[j])**2
            noobj_c_loss = noobj_c_loss + (notchosen[4]-coord_box_right[j][4]*iou[reverse_label[j]][j])**2

    #2


    #3
    noobj_train = train_tensor[reversemask].view(-1,26)
    noobj_train_c1 = noobj_train[:,4]
    noobj_train_c2 = noobj_train[:,9]
    noobj_right = right_tensor[reversemask].view(-1,26)
    noobj_right_c1 = noobj_right[:,4]
    noobj_right_c2 = noobj_right[:,9]
    #result3
    noobj_c_loss= noobj_c_loss + torch.nn.functional.mse_loss(noobj_train_c1,noobj_right_c1,reduction='sum') + torch.nn.functional.mse_loss(noobj_train_c2,noobj_right_c2,reduction='sum')


    #4
    #result4
    class_loss=torch.nn.functional.mse_loss(coord_class_train,coord_class_right,reduction='sum')


    return (lamda_coord*(coord_loss) + obj_conf_loss + lamda_noobj*(noobj_c_loss) + class_loss)/n


"""
z=torch.rand(2,7,7,5)
a=torch.cat((z,z,torch.rand(2,7,7,16)),-1)
b=a
print(trainloss(a,b,5,0.5))
c=torch.tensor([[0.5,0.5,1./7,1./7],[1.,1.,1./7,1./7],[1.,1.,1./7,1./7]])
d=torch.tensor([[1.,1.,1./7,1./7],[1.,1.,1./7,1./7]])
e=calculate_iou(c,d)
print(e)
print(e.max(0))
"""
