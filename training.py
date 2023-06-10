from os import listdir
from os.path import join
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, ToTensor, Grayscale
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
from uitlisze import testing
from RDN_CA import RDN_CA
class L1_Charbonnier_loss(torch.nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss

def calculate_valid_crop_size(crop_size, blocksize):
    return crop_size - (crop_size % blocksize)

def train_hr_transform():
    return Compose([     #combine
        Grayscale(),     #RGB2grey
        ToTensor(),
    ])

class TrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir):
        super(TrainDatasetFromFolder, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir)]
        self.hr_transform = train_hr_transform()

    def __getitem__(self, index):
        try:
            hr_image = self.hr_transform(Image.open(self.image_filenames[index]))
            return hr_image, hr_image
        except:
            hr_image = self.hr_transform(Image.open(self.image_filenames[index+1]))
            return hr_image, hr_image

    def __len__(self):
        return len(self.image_filenames)

train_set = TrainDatasetFromFolder('train_data/SARBud_train_sub')
train_loader = DataLoader(dataset=train_set, num_workers=8, batch_size=8, shuffle=True)


net =RDN_CA(subrate=0.5)
# loss_func = nn.MSELoss()
loss_func = nn.SmoothL1Loss()
if torch.cuda.is_available():
    net.cuda()
    loss_func.cuda()

optimizer = optim.Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.999))
#scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,15], gamma=0.1)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, threshold=0.001)

model_dir = 'RDN_CA_SARBud_0.1_batch_8_lr.pth'
recon_dir = './recon/recon_DN_ship_011.1/'
test_dir = 'Test/SARBud_test'+"/*.*"
loss_values = []
ssim_curve = []
psnr_curve = []
loss_test = []
if __name__=='__main__':
    epochs = 50
    print(epochs)
    print(model_dir)
    for epoch in range(0, epochs):
        if epoch==1:
            for param_group in optimizer.param_groups:
                param_group["lr"] = 1e-4
        train_bar = tqdm(train_loader)
        running_results = {'batch_sizes': 0, 'g_loss': 0, }

        net.train()

        for data, target in train_bar:
            batch_size = data.size(0)
            if batch_size <= 0:
                continue

            running_results['batch_sizes'] += batch_size

            real_img = Variable(target)
            if torch.cuda.is_available():
                real_img = real_img.cuda()
            z = Variable(data)
            if torch.cuda.is_available():
                z = z.cuda()
            fake_img = net(z)
            optimizer.zero_grad()
            g_loss = loss_func(fake_img, real_img)

            g_loss.backward()
            optimizer.step()

            running_results['g_loss'] += g_loss.item() * batch_size

            train_bar.set_description(desc='[%d] Loss_G: %.8f lr: %.8f' % (
                epoch, running_results['g_loss'] / running_results['batch_sizes'], optimizer.param_groups[0]['lr']))
        loss_each_epoch = running_results['g_loss'] / running_results['batch_sizes']

        torch.save(net.state_dict(), model_dir)
        loss_values.append(loss_each_epoch)
        [avg_PSNR,avg_SSIM,avg_loss] = testing(net,model_dir,recon_dir,test_dir)
        psnr_curve.append(avg_PSNR)
        ssim_curve.append(avg_SSIM)
        loss_test.append(avg_loss)
        scheduler.step(loss_each_epoch)
    print('---------------------train finish-------------------------')
    print(model_dir)
    print('loss_train:',loss_values)
    print('psnr_curve:',psnr_curve)
    print('ssim_curve:',ssim_curve)
    #print('loss_test',loss_test)
