
"""
script to train a resnet 50 network only with n epoch
Version 4
plot x y z alpha beta gamma error
plot render after each epoch
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from utils_functions.resnet50 import resnet50
from utils_functions.train_val import train
from utils_functions.cubeDataset import CubeDataset

# device = torch.device('cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()
print(device)

file_name_extension = 'WristwithMultMovingBackground'  # choose the corresponding database to use

batch_size = 10

n_epochs = 10

target_size = (512, 512)

cubes_file = 'Npydatabase/cubes_{}.npy'.format(file_name_extension)
silhouettes_file = 'Npydatabase/sils_{}.npy'.format(file_name_extension)
parameters_file = 'Npydatabase/params_{}.npy'.format(file_name_extension)

fileExtension = 'test' #string to ad at the end of the file

cubeSetName = 'wrist_{}'.format(file_name_extension) #used to describe the document name

date4File = '071019_{}'.format(fileExtension) #mmddyy


cubes = np.load(cubes_file)
sils = np.load(silhouettes_file)
params = np.load(parameters_file)

#  ------------------------------------------------------------------
ratio = 0.9  # 90%training 10%validation
split = int(len(cubes)*0.9)
test_length = 1000

train_im = cubes[:split]  # 90% training
train_sil = sils[:split]
train_param = params[:split]

# plt.imshow(train_sil[0])
# plt.show()

val_im = cubes[split:]  # remaining ratio for validation
val_sil = sils[split:]
val_param = params[split:]

test_im = cubes[:test_length]
test_sil = sils[:test_length]
test_param = params[:test_length]

#  ------------------------------------------------------------------

from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose, Normalize, Lambda


normalize = Normalize(mean=[0.5], std=[0.5])
gray_to_rgb = Lambda(lambda x: x.repeat(3, 1, 1))
transforms = Compose([ToTensor(),  normalize])
train_dataset = CubeDataset(train_im, train_sil, train_param, transforms)
val_dataset = CubeDataset(val_im, val_sil, val_param, transforms)
test_dataset = CubeDataset(test_im, test_sil, test_param, transforms)


# plt.imshow(train_dataset.silhouettes[0])
# plt.show()


#  Note:
#  DataLoader(Dataset,int,bool,int)
#  dataset (Dataset) – dataset from which to load the data.
#  batch_size (int, optional) – how many samples per batch to load (default: 1)
#  shuffle (bool, optional) – set to True to have the data reshuffled at every epoch (default: False).
#  num_workers = n - how many threads in background for efficient loading

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)


for image, sil, param in train_dataloader:

    # print(image[2])
    print(image.size(), param.size()) #torch.Size([batch, 3, 512, 512]) torch.Size([batch, 6])
    im =2
    print(param[im])  # parameter in form tensor([2.5508, 0.0000, 0.0000, 0.0000, 0.0000, 5.0000])

    image2show = image[im]  # indexing random  one image
    print(image2show.size()) #torch.Size([3, 512, 512])
    plt.imshow((image2show * 0.5 + 0.5).numpy().transpose(1, 2, 0))
    plt.show()
    break  # break here just to show 1 batch of data

#  ------------------------------------------------------------------

# for noise in np.arange(0, 1, 0.1):

noise = 0.0
model = resnet50(cifar=True) #use pretrained on imagenet if cifar is true
model = model.to(device)  # transfer the neural net onto the GPU
criterion = nn.MSELoss() # define the loss (MSE, Crossentropy, Binarycrossentropy)

#  ------------------------------------------------------------------

all_Train_losses, all_Test_losses = train(model, train_dataloader, test_dataloader, n_epochs, criterion, date4File, cubeSetName, batch_size, fileExtension, device, noise)

#  ------------------------------------------------------------------

torch.save(model.state_dict(), 'models/{}_FinalModel_train_{}_{}_batchs_{}_epochs_{}_Regr.pth'.format(date4File, cubeSetName, str(batch_size), str(n_epochs), fileExtension))
print('parameters saved')

#  ------------------------------------------------------------------
