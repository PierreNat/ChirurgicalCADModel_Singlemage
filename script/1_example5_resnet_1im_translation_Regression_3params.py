"""
Example 5, regression optimization of MSE loss for 1 image with only translation parameter
"""
import os
import argparse
import glob
from torch.utils.data import Dataset
from scipy.spatial.transform.rotation import Rotation as Rot
import torch.nn as nn
import numpy as np
from skimage.io import imread, imsave
import tqdm
import imageio
import time
import torch
from torchvision.models.resnet import ResNet, Bottleneck
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose, Normalize, Lambda
import matplotlib.pyplot as plt
import math as m
import torch.utils.model_zoo as model_zoo
import neural_renderer as nr
from scipy.misc import imsave
import matplotlib2tikz

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, '3D_objects')
result_dir = os.path.join(current_dir, 'results')


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class CubeDataset(Dataset):
    # code to shape data for the dataloader
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
            sel_sils = torch.from_numpy(sel_sils)

        # squeeze transform sil from tensor shape [6,1,512,512] to shape [6, 512, 512]
        return sel_images, np.squeeze(sel_sils), torch.FloatTensor(sel_params)  # return all parameter in tensor form

    def __len__(self):
        return len(self.images)  # return the length of the dataset

def Myresnet50(filename_obj=None, pretrained=True, cifar = True, modelName='None', **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ModelResNet50( filename_obj=filename_obj)
    if pretrained:
        print('using own pre-trained model')

        if cifar == True:
            pretrained_state = model_zoo.load_url(model_urls['resnet50'])
            model_state = model.state_dict()
            pretrained_state = {k: v for k, v in pretrained_state.items() if
                                k in model_state and v.size() == model_state[k].size()}
            model_state.update(pretrained_state)
            model.load_state_dict(model_state)
            model.eval()

        else:
            model.load_state_dict(torch.load('models/{}.pth'.format(modelName)))
            model.eval()
        print('download finished')
    return model


class ModelResNet50(ResNet):
    def __init__(self, filename_obj=None, filename_init=None, *args, **kwargs):
        super(ModelResNet50, self).__init__(Bottleneck, [3, 4, 6, 3], num_classes=3, **kwargs)

# resnet part
        self.seq1 = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.maxpool,

            self.layer1,
            self.layer2
        )

        self.seq2 = nn.Sequential(
            self.layer3,
            self.layer4,
            self.avgpool,
        )

        self.fc

# render part

        vertices, faces, textures = nr.load_obj(filename_obj, load_texture=True)
        vertices = vertices[None, :, :]  # [num_vertices, XYZ] -> [batch_size=1, num_vertices, XYZ]
        faces = faces[None, :, :]  # [num_faces, 3] -> [batch_size=1, num_faces, 3
        textures = textures[None, :, :]

        self.register_buffer('vertices', vertices)
        self.register_buffer('faces', faces)
        self.register_buffer('textures', textures)

        # ---------------------------------------------------------------------------------
        # extrinsic parameter, link world/object coordinate to camera coordinate
        # ---------------------------------------------------------------------------------

        alpha = np.radians(0)
        beta = np.radians(0)
        gamma = np.radians(0)

        x = 0  # uniform(-2, 2)
        y = 0  # uniform(-2, 2)
        z = 12  # uniform(5, 10) #1000t was done with value between 7 and 10, Rot and trans between 5 10

        resolutionX = 512  # in pixel
        resolutionY = 512
        scale = 1
        f = 35  # focal on lens
        sensor_width = 32  # in mm given in blender , camera sensor type
        pixels_in_u_per_mm = (resolutionX * scale) / sensor_width
        pixels_in_v_per_mm = (resolutionY * scale) / sensor_width
        pix_sizeX = 1 / pixels_in_u_per_mm
        pix_sizeY = 1 / pixels_in_v_per_mm

        Cam_centerX = resolutionX / 2
        Cam_centerY = resolutionY / 2

        batch = vertices.shape[0]

        Rx = np.array([[1, 0, 0],
                       [0, m.cos(alpha), -m.sin(alpha)],
                       [0, m.sin(alpha), m.cos(alpha)]])

        Ry = np.array([[m.cos(beta), 0, m.sin(beta)],
                       [0, 1, 0],
                       [-m.sin(beta), 0, m.cos(beta)]])

        Rz = np.array([[m.cos(gamma), -m.sin(gamma), 0],
                       [m.sin(gamma), m.cos(gamma), 0],
                       [0, 0, 1]])

        #   creaete the rotation camera matrix

        Rzy = np.matmul(Rz, Ry)
        Rzyx = np.matmul(Rzy, Rx)
        R = Rzyx

        t = np.array([x, y, z])  # camera position [x,y, z] 0 0 5

        # ---------------------------------------------------------------------------------
        # intrinsic parameter, link camera coordinate to image plane
        # ---------------------------------------------------------------------------------

        K = np.array([[f / pix_sizeX, 0, Cam_centerX],
                      [0, f / pix_sizeY, Cam_centerY],
                      [0, 0, 1]])  # shape of [nb_vertice, 3, 3]

        K = np.repeat(K[np.newaxis, :, :], batch, axis=0)  # shape of [batch=1, 3, 3]
        R = np.repeat(R[np.newaxis, :, :], batch, axis=0)  # shape of [batch=1, 3, 3]
        t = np.repeat(t[np.newaxis, :], 1, axis=0)  # shape of [1, 3]

        self.K = K
        # self.R = nn.Parameter(torch.from_numpy(np.array(R, dtype=np.float32)))
        self.R = R
        # self.Rx
        # self.Ry
        # self.Rz
        # quaternion notation?
        # -------------------------- working block translation
        self.tx = torch.from_numpy(np.array(x, dtype=np.float32)).cuda()
        self.ty = torch.from_numpy(np.array(y, dtype=np.float32)).cuda()
        self.tz = torch.from_numpy(np.array(z, dtype=np.float32)).cuda()
        self.t =torch.from_numpy(np.array([self.tx, self.ty, self.tz], dtype=np.float32)).unsqueeze(0)
        # self.t = nn.Parameter(torch.from_numpy(np.array([self.tx, self.ty, self.tz], dtype=np.float32)).unsqueeze(0))

        # --------------------------

        # setup renderer
        renderer = nr.Renderer(camera_mode='projection', orig_size=512, K=K, R=R, t=self.t, image_size=512, near=1,
                               far=1000,
                               light_intensity_ambient=1, light_intensity_directional=0, background_color=[0, 0, 0],
                               light_color_ambient=[1, 1, 1], light_color_directional=[1, 1, 1],
                               light_direction=[0, 1, 0])

        self.renderer = renderer

    def forward(self, x):
        x = self.seq1(x)
        x = self.seq2(x)
        params = self.fc(x.view(x.size(0), -1))
        print('computed parameters are {}'.format(params))
        return params

# ---------------------------------------------------------------------------------
# make Gif
# ---------------------------------------------------------------------------------
def make_gif(filename):
    with imageio.get_writer(filename, mode='I') as writer:
        for filename in sorted(glob.glob('/tmp/_tmp_*.png')):
            writer.append_data(imread(filename))
            os.remove(filename)
    writer.close()

# ---------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------
def main():
    start = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()
    print(device)

    file_name_extension = 'Translation_im1'  # choose the corresponding database to use

    cubes_file = 'Npydatabase/wrist_{}.npy'.format(file_name_extension)
    silhouettes_file = 'Npydatabase/sils_{}.npy'.format(file_name_extension)
    parameters_file = 'Npydatabase/params_{}.npy'.format(file_name_extension)

    wrist = np.load(cubes_file)
    sils = np.load(silhouettes_file)
    params = np.load(parameters_file)

    train_im = wrist  # 90% training
    train_sil = sils
    train_param = params

    normalize = Normalize(mean=[0.5], std=[0.5])
    transforms = Compose([ToTensor(), normalize])
    train_dataset = CubeDataset(train_im, train_sil, train_param, transforms)


    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1)

    # # check to iterate inside the test dataloader
    # for image, sil, param in train_dataloader:
    #
    #     # print(image[2])
    #     print(image.size(), param.size()) #torch.Size([batch, 3, 512, 512]) torch.Size([batch, 6])
    #     im =0
    #     print(param[im])  # parameter in form tensor([2.5508, 0.0000, 0.0000, 0.0000, 0.0000, 5.0000])
    #
    #     image2show = image[im]  # indexing random  one image
    #     print(image2show.size()) #torch.Size([3, 512, 512])
    #     plt.imshow((image2show * 0.5 + 0.5).numpy().transpose(1, 2, 0))
    #     plt.show()
    #     break  # break here just to show 1 batch of data


    count = 0
    losses = []
    a = []
    b = []
    c = []
    tx = []
    ty = []
    tz = []
    #ground value to be plotted on the graph as line
    alpha_GT = np.array( m.degrees(params[0,0]))
    beta_GT =  np.array(m.degrees(params[0,1]))
    gamma_GT =  np.array(m.degrees(params[0,2]))#angle in degrer
    tx_GT =  np.array(params[0,3])
    ty_GT = np.array(params[0,4])
    tz_GT = np.array(params[0,5])


    iterations = 100
    parser = argparse.ArgumentParser()
    parser.add_argument('-io', '--filename_obj', type=str, default=os.path.join(data_dir, 'wrist.obj'))
    parser.add_argument('-or', '--filename_output', type=str, default=os.path.join(result_dir, '{}_regression_animation.gif'.format(file_name_extension)))
    parser.add_argument('-mr', '--make_reference_image', type=int, default=0)
    parser.add_argument('-g', '--gpu', type=int, default=0)
    args = parser.parse_args()

    # resnet50 = models.resnet50(pretrained=True)

    model = Myresnet50(filename_obj=args.filename_obj)
    # model = Model(args.filename_obj, args.filename_ref)

    model.to(device)

    model.train(True)

    Lr_start = 0.0001
    decreaseat = 40
    lr = Lr_start

    loop = tqdm.tqdm(range(iterations))
    for i in loop:

        for image, silhouette, parameter in train_dataloader:
            image = image.to(device)

            parameter = parameter.to(device)
            # print(parameter)
            silhouette = silhouette.to(device)
            params = model(image)
            model.t = params
            bool_first = True
            # first_
            print(model.t)

             # regression between computed and ground truth
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            loss = nn.MSELoss()(params, parameter[0, 3:6]).to(device)
            if (i % decreaseat == 0 and i>2):
                lr = lr / 10
                print('update lr, is now {}'.format(lr))

            print('loss is {}'.format(loss))


            # ref = np.squeeze(model.image_ref[None, :, :]).cpu()
            # image = image.detach().cpu().numpy().transpose((1, 2, 0))
            # image = np.squeeze((image * 255)).astype(np.uint8) # change from float 0-1 [512,512,1] to uint8 0-255 [512,512]
            # fig = plt.figure()
            # fig.add_subplot(1, 2, 1)
            # plt.imshow(image, cmap='gray')
            # fig.add_subplot(1, 2, 2)
            # plt.imshow(ref, cmap='gray')
            # plt.show()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.detach().cpu().numpy())
            cp_x = np.round(((model.t).detach().cpu().numpy())[0, 0], 2)
            cp_y = np.round(((model.t).detach().cpu().numpy())[0, 1], 2)
            cp_z = np.round(((model.t).detach().cpu().numpy())[0, 2], 2)

            cp_rotMat = (model.R) #cp_rotMat = (model.R).detach().cpu().numpy()
            r = Rot.from_dcm(cp_rotMat)
            r_euler = r.as_euler('xyz', degrees=True)

            a.append(abs(r_euler[0, 0])) #        a.append(abs(r_euler[0,0] ))
            b.append(abs(r_euler[0, 1]))
            c.append(abs(r_euler[0, 2]))

            tx.append(cp_x)
            ty.append(cp_y)
            tz.append(cp_z) #z axis value

            images, _, _ = model.renderer(model.vertices, model.faces, torch.tanh(model.textures),t= model.t )

            img = images.detach().cpu().numpy()[0].transpose(1,2,0)

            if(i == iterations-1):

                imgGT = image.squeeze()  # float32 from 0-1
                imgGT = imgGT.detach().cpu()
                imgGT = (imgGT * 0.5 + 0.5).numpy().transpose(1, 2, 0)
                # imgGT = (imgGT * 255).astype(np.uint8)  # cast from float32 255.0 to 255 uint8

                a = plt.subplot(1, 2, 1)
                plt.imshow(imgGT)
                a.set_title('Ground truth \ntx {}\nty {}\ntz {}'.format(tx_GT, ty_GT, tz_GT))
                plt.xticks([0, 512])
                plt.yticks([])
                a = plt.subplot(1, 2,2)
                plt.imshow(img)
                a.set_title('Regression \ntx {:.3f}\nty {:.3f}\ntz {:.3f}'.format(cp_x, cp_y, cp_z))
                plt.xticks([0, 512])
                plt.yticks([])

                plt.savefig('results/Final_regression_translation_{}iterations_{}.png'.format(iterations, file_name_extension),  bbox_inches = 'tight', pad_inches = 0.05)


            imsave('/tmp/_tmp_%04d.png' % i, img)
            loop.set_description('Optimizing (loss %.4f)' % loss.data)
            count = count +1

    end = time.time()
    exectime = round((end - start), 2) #format in minute
    print('time elapsed is: {} sec'.format(exectime))


    make_gif(args.filename_output)
    fig, (p1, p2) = plt.subplots(2, figsize=(15,10)) #largeur hauteur
    fig.suptitle("Regression for 1 image, {} epochs in {} sec, 3 translation parameters \n lr={} and decrease each {} iterations".format(iterations,exectime, Lr_start, decreaseat), fontsize=14)

    p1.plot(np.arange(count), losses, label="Global Loss")
    p1.set( ylabel='MSE Loss')
    p1.set_ylim([0, 1])
    p1.set(xlabel='Iterations')
    # Place a legend to the right of this smaller subplot.
    p1.legend()

    p2.plot(np.arange(count), tx, label="x values", color = 'g' )
    p2.axhline(y=tx_GT, color = 'g', linestyle= '--' )
    p2.plot(np.arange(count), ty, label="y values", color = 'y')
    p2.axhline(y=ty_GT, color = 'y', linestyle= '--' )
    p2.plot(np.arange(count), tz, label="z values", color = 'b')
    p2.axhline(y=tz_GT, color = 'b', linestyle= '--' )

    p2.set(ylabel='Translation value')
    p2.set_ylim([-5, 10])
    p2.set(xlabel='Iterations')
    p2.legend()



    fig.savefig('results/regression_1image_Translation_3params_{}.pdf'.format(file_name_extension), bbox_inches = 'tight', pad_inches = 0.05)
    fig.savefig('results/regression_1image_Translation_3params_{}.png'.format(file_name_extension), bbox_inches = 'tight', pad_inches = 0.05)
    matplotlib2tikz.save("results/regression_1image_Translation_3params_{}.tex".format(file_name_extension))
    plt.show()
if __name__ == '__main__':
    main()