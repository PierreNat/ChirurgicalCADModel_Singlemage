"""
Example 4. Finding camera parameters.
"""
import os
import argparse
import glob
from torch.utils.data import Dataset
from scipy.spatial.transform.rotation import Rotation as Rot
import torch
import torch.nn as nn
import numpy as np
from skimage.io import imread, imsave
import tqdm
import imageio
import time
from torch.autograd import Variable
import torch
import torchvision.models as models
from torchvision.models.resnet import ResNet, Bottleneck
import torchvision.models as models
import torchgeometry as tgm #from https://torchgeometry.readthedocs.io/en/v0.1.2/_modules/torchgeometry/core/homography_warper.html
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose, Normalize, Lambda
import matplotlib.pyplot as plt
import math as m
import torch.utils.model_zoo as model_zoo
import neural_renderer as nr
from scipy.misc import imsave

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, 'data')


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

def Myresnet50(filename_obj=None, filename_ref=None, filename_init=None, pretrained=True, cifar = True, modelName='None', **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ModelResNet50( filename_obj=filename_obj, filename_ref=filename_ref, filename_init= filename_init)
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
    def __init__(self, filename_obj=None, filename_ref=None, filename_init=None, *args, **kwargs):
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

        # load reference image
        image_ref = torch.from_numpy((imread(filename_ref).max(-1) != 0).astype(np.float32))
        self.register_buffer('image_ref', image_ref)
         # create image to init the resnet weight at first
        image_init = torch.from_numpy((imread(filename_init).max(-1) != 0).astype(np.float32))
        self.register_buffer('image_init', image_init)

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

    file_name_extension = 'wrist1im'  # choose the corresponding database to use

    cubes_file = 'Npydatabase/cubes_{}.npy'.format(file_name_extension)
    silhouettes_file = 'Npydatabase/sils_{}.npy'.format(file_name_extension)
    parameters_file = 'Npydatabase/params_{}.npy'.format(file_name_extension)

    wrist = np.load(cubes_file)
    sils = np.load(silhouettes_file)
    params = np.load(parameters_file)

    train_im = wrist  # 90% training
    train_sil = sils
    train_param = params

    normalize = Normalize(mean=[0.5], std=[0.5])
    gray_to_rgb = Lambda(lambda x: x.repeat(3, 1, 1))
    transforms = Compose([ToTensor(), normalize])
    train_dataset = CubeDataset(train_im, train_sil, train_param, transforms)


    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1)


    count = 0
    losses = []
    a = []
    b = []
    c = []
    tx = []
    ty = []
    tz = []
    #ground value to be plotted on the graph as line
    alpha_GT = 0
    beta_GT = 0
    gamma_GT = 0 #angle in degrer
    tx_GT = 0
    ty_GT = 0
    tz_GT = 5

    iterations = 1000
    file_name_extension = 'regression'
    parser = argparse.ArgumentParser()
    parser.add_argument('-io', '--filename_obj', type=str, default=os.path.join(data_dir, 'wrist.obj'))
    parser.add_argument('-ir', '--filename_ref', type=str, default=os.path.join(data_dir, 'example5_refT.png')) #image result to target
    parser.add_argument('-in', '--filename_init', type=str, default=os.path.join(data_dir, 'example5_inT.png')) # image to init resnet with regression
    parser.add_argument('-or', '--filename_output', type=str, default=os.path.join(data_dir, 'example5_resultT_regression.gif'))
    parser.add_argument('-mr', '--make_reference_image', type=int, default=0)
    parser.add_argument('-g', '--gpu', type=int, default=0)
    args = parser.parse_args()

    # resnet50 = models.resnet50(pretrained=True)

    model = Myresnet50(filename_obj=args.filename_obj, filename_ref=args.filename_ref, filename_init= args.filename_init)
    # model = Model(args.filename_obj, args.filename_ref)

    model.to(device)

    model.train(True)
    loop = tqdm.tqdm(range(iterations))
    for i in loop:

        for image, silhouette, parameter in train_dataloader:
            image = image.to(device)
            parameter = parameter.to(device)
            print(parameter)
            silhouette = silhouette.to(device)
            params = model(image)
            model.t = params
            bool_first = True
            # first_
            print(model.t)

             # regression between computed and ground truth

            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
            loss = nn.MSELoss()(params, parameter[0, 3:6]).to(device)


            print('loss is {}'.format(loss))
            ref = np.squeeze(model.image_ref[None, :, :]).cpu()
            image = image.detach().cpu().numpy().transpose((1, 2, 0))
            image = np.squeeze((image * 255)).astype(np.uint8) # change from float 0-1 [512,512,1] to uint8 0-255 [512,512]
            fig = plt.figure()
            fig.add_subplot(1, 2, 1)
            plt.imshow(image, cmap='gray')
            fig.add_subplot(1, 2, 2)
            plt.imshow(ref, cmap='gray')
            plt.show()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.detach().cpu().numpy())
            # print(((model.K).detach().cpu().numpy()))
            cp_x = ((model.t).detach().cpu().numpy())[0, 0]
            cp_y = ((model.t).detach().cpu().numpy())[0, 1]
            cp_z = ((model.t).detach().cpu().numpy())[0, 2]

            cp_rotMat = (model.R) #cp_rotMat = (model.R).detach().cpu().numpy()
            r = Rot.from_dcm(cp_rotMat)
            r_euler = r.as_euler('xyz', degrees=True)

            # print(r_euler)
            # a.append(abs(r_euler[0,0] - alpha_GT))
            # b.append(abs(r_euler[0,1] - beta_GT))
            # c.append(abs(r_euler[0,2] - gamma_GT))

            a.append(abs(r_euler[0, 0])) #        a.append(abs(r_euler[0,0] ))
            b.append(abs(r_euler[0, 1]))
            c.append(abs(r_euler[0, 2]))

            # print (r_euler[0,2], r_euler[0,2]% 180)

            # tx.append(abs(cp_x - tx_GT))
            # ty.append(abs(cp_y - ty_GT))
            # tz.append(abs(cp_z)) #z axis error

            tx.append(abs(cp_x))
            ty.append(abs(cp_y))
            tz.append(abs(cp_z)) #z axis value

            images, _, _ = model.renderer(model.vertices, model.faces, torch.tanh(model.textures),t= model.t )

            image = images.detach().cpu().numpy()[0].transpose(1,2,0)
            # plt.imshow(image)
            # plt.show()
            imsave('/tmp/_tmp_%04d.png' % i, image)
            loop.set_description('Optimizing (loss %.4f)' % loss.data)
            count = count +1
            # if loss.item() == 180:
            #     break

    make_gif(args.filename_output)
    fig, (p1, p2, p3) = plt.subplots(3, figsize=(15,10)) #largeur hauteur

    p1.plot(np.arange(count), losses, label="Global Loss")
    p1.set( ylabel='BCE Loss')
    p1.set_ylim([0, 1])
    # Place a legend to the right of this smaller subplot.
    p1.legend()

    p2.plot(np.arange(count), tx, label="x values")
    p2.axhline(y=tx_GT)
    p2.plot(np.arange(count), ty, label="y values")
    p2.axhline(y=ty_GT)
    p2.plot(np.arange(count), tz, label="z values")
    p2.axhline(y=tz_GT)

    p2.set(ylabel='Translation value')
    p2.set_ylim([0, 10])
    p2.legend()

    p3.plot(np.arange(count), a, label="alpha values")
    p3.axhline(y=alpha_GT)
    p3.plot(np.arange(count), b, label="beta values")
    p3.axhline(y=beta_GT)
    p3.plot(np.arange(count), c, label="gamma values")
    p3.axhline(y=gamma_GT)

    p3.set(xlabel='iterations', ylabel='Rotation value')
    p3.legend()

    fig.savefig('images/ex5plot_{}.pdf'.format(file_name_extension))
    import matplotlib2tikz

    matplotlib2tikz.save("images/ex5plot_{}.tex".format(file_name_extension))

    plt.show()
    plt.show()
    end = time.time()
    print('time elapsed is: {} min'.format((end - start)/60))

if __name__ == '__main__':
    main()