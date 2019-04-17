import torch
import torch.utils.data as DATA
from torch.autograd import Variable

import loss
import models
import loaddata

#reading data

class dataset_loaded(DATA.Dataset):
    def __init__(self):
        self.img,self.target=loaddata.read_training_data()
    def __getitem__(self,idx):
        img_3_448_448=torch.stack((self.img[idx][:,:,0],self.img[idx][:,:,1],self.img[idx][:,:,2]))
        return img_3_448_448.float(),self.target[idx]
    def __len__(self):
        return self.img.size()[0]

class test_dataset_loaded(DATA.Dataset):
    def __init__(self):
        self.img,self.target=loaddata.read_val_data()
    def __getitem__(self,idx):
        img_3_448_448=torch.stack((self.img[idx][:,:,0],self.img[idx][:,:,1],self.img[idx][:,:,2]))
        return img_3_448_448.float(),self.target[idx]
    def __len__(self):
        return self.img.size()[0]

def main():
    epoch=50
    batchsize=16
    lr=0.0001

    net=models.Yolov1_vgg16bn(pretrained=True)

    net.cuda()
    #
    net.load_state_dict(torch.load("saved_model29.pth"))
    #
    net.train()
    print("model loaded")
    """
    training_img=DATA.Dataset(loaddata.read_training_data())
    training_img_loader=DATA.DataLoader(training_img,batch_size=batchsize,shuffle=True,num_workers=4)
    print("OK")
    training_target=DATA.TensorDataset(loaddata.read_training_data()[1])
    training_target_loader=DATA.DataLoader(training_target,batch_size=batchsize,shuffle=True)
    print()
    """
    training_dataset=dataset_loaded()
    training_data_loader=DATA.DataLoader(training_dataset,batch_size=batchsize,shuffle=True,num_workers=0)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    """#
    testing_dataset=test_dataset_loaded()
    testing_data_loader=DATA.DataLoader(testing_dataset,batch_size=batchsize,shuffle=False)
    """#

    print("loaded")
    for e in range(29,epoch):

        if e == 30:
            lr=0.0001
        if e == 40:
            lr=0.00001
        runningloss=0.
        test_loss=0.
        count=0

        for i,(images,targets) in enumerate(training_data_loader):

            images=Variable(images)
            images=images.cuda()
            targets=Variable(targets)
            targets=targets.cuda()
            optimizer.zero_grad()

            outputs=net(images)

            batch_loss=loss.trainloss(outputs,targets,5.,0.5)

            batch_loss.backward()
            optimizer.step()

            runningloss=runningloss+batch_loss
            count=count+1

        """#
        for j,(test_images,test_targets) in enumerate(testing_data_loader):

            test_images=Variable(test_images)
            test_images=test_images.cuda()
            test_targets=Variable(test_targets)
            test_targets=test_targets.cuda()

            test_outputs=net(test_images)

            test_loss=test_loss+loss.trainloss(test_outputs.cuda(),test_targets,5.,0.5)
        """#

        torch.save(net.state_dict(),('train_model%02d.pth'%e))
        print("epoch ",e," test loss:",test_loss/count)
        print("epoch ",e," loss:",runningloss/count)
    #torch.save(net.state_dict(),'saved_model.pth')
if __name__ == '__main__':
    main()
