"""
Example 4. Finding camera parameters.
"""
import os
import argparse
import glob
from scipy.spatial.transform.rotation import Rotation as Rot
import torch
import torch.nn as nn
import numpy as np
from skimage.io import imread, imsave
import tqdm
import imageio
import matplotlib.pyplot as plt
import math as m

import neural_renderer as nr
from scipy.misc import imsave

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, 'data')

class Model(nn.Module):
    def __init__(self, filename_obj, filename_ref=None):
        super(Model, self).__init__()
        # load .obj
        texture_size = 2
        # vertices, faces = nr.load_obj(filename_obj)
        vertices, faces = nr.load_obj(filename_obj)
        vertices = vertices[None, :, :]  # [num_vertices, XYZ] -> [batch_size=1, num_vertices, XYZ]
        faces = faces[None, :, :]  # [num_faces, 3] -> [batch_size=1, num_faces, 3
        # create texture [batch_size=1, num_faces, texture_size, texture_size, texture_size, RGB]
        textures = torch.ones(1, faces.shape[1], texture_size, texture_size, texture_size, 3,
                              dtype=torch.float32).cuda()

        self.register_buffer('vertices', vertices)
        self.register_buffer('faces', faces)
        self.register_buffer('textures', textures)

        # self.register_buffer('vertices', vertices[None, :, :])
        # self.register_buffer('faces', faces[None, :, :])
        #
        # # create textures
        # texture_size = 2
        # textures = torch.ones(1, self.faces.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32)
        # self.register_buffer('textures', textures)
        #


        # camera parameters (elevation angle, distance)
        # self.camera_position = nn.Parameter(torch.from_numpy(np.array([6, 10, -14], dtype=np.float32)))

        # load reference image
        image_ref = torch.from_numpy((imread(filename_ref).max(-1) != 0).astype(np.float32))
        self.register_buffer('image_ref', image_ref)

        # ---------------------------------------------------------------------------------
        # extrinsic parameter, link world/object coordinate to camera coordinate
        # ---------------------------------------------------------------------------------

        alpha = 0
        beta = 0
        gamma = 90
        x = 0  # uniform(-2, 2)
        y = 0  # uniform(-2, 2)
        z = 7  # uniform(5, 10) #1000t was done with value between 7 and 10, Rot and trans between 5 10

        resolutionX = 256  # in pixel
        resolutionY = 256
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

        R = np.matmul(Rx, Ry)
        R = np.matmul(R, Rz)

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

        self.K = nn.Parameter(torch.from_numpy(np.array(K, dtype=np.float32)))
        self.R = nn.Parameter(torch.from_numpy(np.array(R, dtype=np.float32)))
        self.t = nn.Parameter(torch.from_numpy(np.array(t, dtype=np.float32)))

        # ---------------------------------------------------------------------------------
        # put in renderer allowed format batch_sizex3x3
        # ---------------------------------------------------------------------------------


        #
        # self.camera_position = nn.Parameter(torch.from_numpy(np.array([6, 10, -14], dtype=np.float32)))

        # setup renderer
        renderer = nr.Renderer(camera_mode='projection', K=self.K, R=self.R, t=self.t, image_size=512, near=1, far=1000, light_intensity_ambient=0.5, light_intensity_directional=0.5,
                 light_color_ambient=[1,1,1], light_color_directional=[1,1,1],
                 light_direction=[0,1,0])
        # renderer.K = self.K
        # renderer.R = self.R
        # renderer.t = self.t
        # renderer = nr.Renderer(camera_mode='look_at', image_size=256)
        # renderer.eye = self.camera_position
        self.renderer = renderer

    def forward(self):
        image = self.renderer(self.vertices, self.faces, mode='silhouettes')
        # image has size [1,256,256]
        # image self size [256,256]

        loss = nn.BCELoss()(image, self.image_ref[None, :, :])
        # loss = torch.sum((image - self.image_ref[None, :, :]) ** 2)
        #
        # ref = np.squeeze(self.image_ref[None, :, :]).cpu()
        # image = image.detach().cpu().numpy().transpose((1, 2, 0))
        # image = np.squeeze((image * 255)).astype(np.uint8) # change from float 0-1 [512,512,1] to uint8 0-255 [512,512]
        # fig = plt.figure()
        # fig.add_subplot(1, 2, 1)
        # plt.imshow(image, cmap='gray')
        # fig.add_subplot(1, 2, 2)
        # plt.imshow(ref, cmap='gray')
        # plt.show()


        return loss


def make_gif(filename):
    with imageio.get_writer(filename, mode='I') as writer:
        for filename in sorted(glob.glob('/tmp/_tmp_*.png')):
            writer.append_data(imread(filename))
            os.remove(filename)
    writer.close()


# def make_reference_image(filename_ref, filename_obj):
#     model = Model(filename_obj)
#     model.cuda()
#     model.renderer.eye = nr.get_points_from_angles(2.732, 45, -15)
#     images, _, _ = model.renderer.render(model.vertices, model.faces, torch.tanh(model.textures))
#     image = images.detach().cpu().numpy()[0]
#     imsave(filename_ref, image)


def main():
    count = 0
    losses = []
    a = []
    b = []
    c = []
    tx = []
    ty = []
    tz = []
    alpha_GT = 0
    beta_GT = 0
    gamma_GT = 100 #angle in degrer
    tx_GT = 0
    ty_GT = 0
    tz_GT = 6
    parser = argparse.ArgumentParser()
    parser.add_argument('-io', '--filename_obj', type=str, default=os.path.join(data_dir, 'wrist.obj'))
    parser.add_argument('-ir', '--filename_ref', type=str, default=os.path.join(data_dir, 'example5_ref.png'))
    parser.add_argument('-or', '--filename_output', type=str, default=os.path.join(data_dir, 'example5_result.gif'))
    parser.add_argument('-mr', '--make_reference_image', type=int, default=0)
    parser.add_argument('-g', '--gpu', type=int, default=0)
    args = parser.parse_args()
    # loss_function2 = nn.BCELoss()
    # if args.make_reference_image:
    #     make_reference_image(args.filename_ref, args.filename_obj)

    model = Model(args.filename_obj, args.filename_ref)
    model.cuda()

    # optimizer = chainer.optimizers.Adam(alpha=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    loop = tqdm.tqdm(range(500))
    for i in loop:
        optimizer.zero_grad()
        loss = model()
        loss.backward()
        optimizer.step()

        losses.append(loss.detach().cpu().numpy())
        cp_x = ((model.t).detach().cpu().numpy())[0, 0]
        cp_y = ((model.t).detach().cpu().numpy())[0, 1]
        cp_z = ((model.t).detach().cpu().numpy())[0,2]
        # cp_rotMat = (model.R).detach().cpu().numpy()
        # r = Rot.from_dcm(cp_rotMat)
        # r_euler = r.as_euler('xyz', degrees=True)
        # print(r_euler)
        # a.append(abs(r_euler[0,0] - alpha_GT))
        # b.append(abs(r_euler[0,1] - beta_GT))
        # c.append(abs(r_euler[0,2] - gamma_GT))
        tx.append(abs(cp_x - tx_GT))
        ty.append(abs(cp_y - ty_GT))
        tz.append(abs(cp_z - tz_GT)) #z axis error

        images, _, _ = model.renderer(model.vertices, model.faces, torch.tanh(model.textures))
        image = images.detach().cpu().numpy()[0].transpose(1,2,0)
        imsave('/tmp/_tmp_%04d.png' % i, image)
        loop.set_description('Optimizing (loss %.4f)' % loss.data)
        count = count +1
        # if loss.item() < 0.015:
        #     break

    make_gif(args.filename_output)
    fig, (p1, p2, p3) = plt.subplots(3,sharex=True)

    p1.plot(np.arange(count), losses, label="Global Loss")
    p1.set( ylabel='BCE Loss')

    # Place a legend to the right of this smaller subplot.
    p1.legend()

    p2.plot(np.arange(count), tx, label="x values")
    p2.plot(np.arange(count), ty, label="y values")
    p2.plot(np.arange(count), tz, label="z values")

    p2.set(xlabel='iterations', ylabel='Absolute error Translation [mm]')
    p2.legend()

    # p3.plot(np.arange(count), a, label="alpha values")
    # p3.plot(np.arange(count), b, label="beta values")
    # p3.plot(np.arange(count), c, label="gamma values")
    #
    # p3.set(xlabel='iterations', ylabel='Absolute error Rotation [deg]')
    # p3.legend()



    plt.show()



if __name__ == '__main__':
    main()