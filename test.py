import torch
import torch.utils.data as DATA
from torch.autograd import Variable

import loss
import models
import loaddata


#reading data

class dataset_loaded(DATA.Dataset):
    def __init__(self):
        self.img,self.target=loaddata.read_val_data()
    def __getitem__(self,idx):
        img_3_448_448=torch.stack((self.img[idx][:,:,0],self.img[idx][:,:,1],self.img[idx][:,:,2]))
        return img_3_448_448.float(),self.target[idx]
    def __len__(self):
        return self.img.size()[0]

runningloss=0.
count=0
net=models.Yolov1_vgg16bn(pretrained=True)
net.load_state_dict(torch.load('train_model45.pth'))
net.cuda()

training_dataset=dataset_loaded()
training_data_loader=DATA.DataLoader(training_dataset,batch_size=16,shuffle=True)

print("loaded")
for i,(images,targets) in enumerate(training_data_loader):
    images=images.cuda()
    tartargets.cuda()
    outputs=net(images)

    runningloss=runningloss+loss.trainloss(outputs,targets,5.,0.5)
    count=count+1
    print(runningloss/(count))
print(runningloss/count)
