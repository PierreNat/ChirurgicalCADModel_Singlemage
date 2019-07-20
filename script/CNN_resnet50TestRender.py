"""
script to train a resnet 50 network only with n epoch
Version 4
render is done during the computation beside the regression
"""

import time
import torch
import torch.nn as nn
import numpy as np
import tqdm
import  matplotlib
import matplotlib.pyplot as plt

from utils_functions.render1item import render_1_image
from utils_functions.resnet50 import resnet50
from utils_functions.testRender import testRenderResnet


# device = torch.device('cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()
print(device)

# modelName = 'Best_Model_translation/070119_Wrist_test_TempModel_train_cubes_wrist_10000_t_batchsOf7img_0.0%noise_epochs_n2_Wrist_test_RenderRegr'
# modelName = '070519_part3_30epochs_TempModel_train_cubes_wrist_10000_Rt_batchsOf7img_0.0%noise_epochs_n1_part3_30epochs_RenderRegr'
# modelName = 'Best_Model_RealBackground/070619_Ubelix_realBackgroundtest_render_part1_TempModel_train_cubes_WristwithBackground_batchsOf20img_0.0%noise_epochs_n15_Ubelix_realBackgroundtest_render_part1_RenderRegr'
modelName = 'Best_Model_RealMultipleMovingBackground/Ubelix_071519_Ubelix_WristwithMultMovingBackground_renderMSE_FinalModel_train_cubes_WristwithMultMovingBackground_20batchs_13epochs_Noise0.0_Ubelix_WristwithMultMovingBackground_renderMSE_Render'

# file_name_extension = 'wrist_10000_t'
# file_name_extension = 'wrist_10000_Rt'
file_name_extension ='WristwithBackground'
file_name_extension = 'WristwithMovingBackground'
file_name_extension = 'WristwithMultMovingBackground'

cubes_file = 'Npydatabase/cubes_{}.npy'.format(file_name_extension)
silhouettes_file = 'Npydatabase/sils_{}.npy'.format(file_name_extension)
parameters_file = 'Npydatabase/params_{}.npy'.format(file_name_extension)

target_size = (512, 512)
obj_name = 'wrist'

cubes = np.load(cubes_file)
sils = np.load(silhouettes_file)
params = np.load(parameters_file)


#  ------------------------------------------------------------------
test_length = 100
batch_size = 5

test_im = cubes[:test_length]
test_sil = sils[:test_length]
test_param = params[:test_length]

#  ------------------------------------------------------------------

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose, Normalize, Lambda

class CubeDataset(Dataset):
    # write your code
    def __init__(self, images, silhouettes, parameters, transform=None):
        self.images = images.astype(np.uint8)  # our image
        self.silhouettes = silhouettes.astype(np.uint8)  # our related parameter
        self.parameters = parameters.astype(np.float32)
        self.transform = transform

    def __getitem__(self, index):
        # Anything could go here, e.g. image loading from file or a different structure
        # must return image and center
        sel_images = self.images[index].astype(np.float32) / 255
        sel_sils = self.silhouettes[index]
        sel_params = self.parameters[index]

        if self.transform is not None:
            sel_images = self.transform(sel_images)
            sel_sils = self.transform(sel_sils)

        return sel_images, sel_images, torch.FloatTensor(sel_params)  # return all parameter in tensor form

    def __len__(self):
        return len(self.images)  # return the length of the dataset
#  ------------------------------------------------------------------


normalize = Normalize(mean=[0.5], std=[0.5])

transforms = Compose([ ToTensor(),  normalize])

test_dataset = CubeDataset(test_im, test_sil, test_param, transforms)

test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)


#  ------------------------------------------------------------------

# # check to iterate inside the test dataloader
# for image, sil, param in test_dataloader:
#
#     # print(image[2])
#     print(image.size(), param.size()) #torch.Size([batch, 3, 512, 512]) torch.Size([batch, 6])
#     im =2
#     print(param[im])  # parameter in form tensor([2.5508, 0.0000, 0.0000, 0.0000, 0.0000, 5.0000])
#
#     image2show = image[im]  # indexing random  one image
#     print(image2show.size()) #torch.Size([3, 512, 512])
#     plt.imshow((image2show * 0.5 + 0.5).numpy().transpose(1, 2, 0))
#     plt.show()
#     break  # break here just to show 1 batch of data


#  ------------------------------------------------------------------


model = resnet50(cifar=False, modelName=modelName) #train with the saved model from the training script
model = model.to(device)  # transfer the neural net onto the GPU
criterion = nn.MSELoss()

#  ------------------------------------------------------------------

# test the model
print("Start timer")
start_time = time.time()
parameters, predicted_params, test_losses, al, bl, gl, xl, yl, zl = testRenderResnet(model, test_dataloader, criterion, file_name_extension, device, obj_name)
print("computing prediction done in  {} seconds ---".format(time.time() - start_time))

#  ------------------------------------------------------------------
# display computed parameter against ground truth


obj_name = 'wrist'
ncols = 5
nrows = 2
Gt = []
Rdr = []
nb_im =5

fig = plt.figure()


# loop = tqdm.tqdm(range(0,nb_im))
for i in range(0, nb_im):
    randIm = i+1 #select a random image
    print('computed parameter_{}: '.format(i))
    print(predicted_params[randIm])
    print('ground truth parameter_{}: '.format(i))
    print(params[randIm])
    print('angle and translation error for {}: '.format(i))
    loss_angle = (predicted_params[randIm][0:3] - params[randIm][0:3])
    loss_translation = (predicted_params[randIm][3:6]-params[randIm][3:6])
    print(loss_angle, loss_translation)
    # print('error {} degree and {} meter '.format(np.rad2deg(predicted_params[randIm][0:3]-params[randIm][0:3]), predicted_params[randIm][3:6]-params[randIm][3:6]))


    im = render_1_image(obj_name, torch.from_numpy(predicted_params[randIm]))  # create the dataset

    Gt.append(test_im[randIm])
    Rdr.append(im)

    a = plt.subplot(2, nb_im, i+1)
    plt.imshow(test_im[randIm])
    a.set_title('GT {}'.format(i))
    plt.xticks([0, 500])
    plt.yticks([])
    a = plt.subplot(2, nb_im, i+1+nb_im)
    plt.imshow(im)
    a.set_title('Rdr {}'.format(i))
    plt.xticks([0, 500])
    plt.yticks([])

    # plt.subplot(2, nb_im, i+1)
    # plt.imshow(test_im[randIm])
    # plt.title('GT {}'.format(i))
    #
    # plt.subplot(2, nb_im, i+1+nb_im)
    # plt.imshow(im)
    # plt.title('Rdr {}'.format(i))


print('finish')


plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.1, hspace=None)
plt.tight_layout()
plt.savefig("image/GroundtruthVsRenderTestRt_realMultMovingbackground_rend2.png")
plt.close(fig)


