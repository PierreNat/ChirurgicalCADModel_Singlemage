import numpy as np
import tqdm
import torch
import matplotlib.pyplot as plt

def lossBtwSils(GT_Sils,ComputedSils, lossfunction, plot=False):


    #NOT WORKING ANY MORE, NO GRAD INFORMATION KEPT

    nbrOfSil = np.shape(GT_Sils)[0]
    gt = GT_Sils #selection of ground truth segmentation
    cp = ComputedSils #selection of computed segmentation
    # cp[1] = gt[1] #simulation of a perfect match
    # plt.imshow(gt.cpu(), cmap='gray')
    # plt.show()
    # print(GD_Sils.max(), GD_Sils.min())
    # print(ComputedSils.max(), ComputedSils.min())
    input = (gt.cpu()).type(torch.FloatTensor)/255.
    target = (cp.cpu()).type(torch.FloatTensor)/255.
    # plt.imshow(gtfloat, cmap='gray')
    # plt.show()
    # print(gtfloat.max(), gtfloat.min())

    # input.requires_grad = True
    print(input.requires_grad)
    lossfunction.reduction = 'none'

    temp_loss = lossfunction(input.double(), target.double())
    lossfunction.reduction = 'mean'
    mean_loss = lossfunction(input.double(), target.double())

    if plot:
        for i in range(0, nbrOfSil):
            plt.subplot(1, nbrOfSil, i + 1)
            plt.imshow(temp_loss[i].detach().numpy(), cmap='gray')
    # plt.imshow(temp_loss[0].detach().numpy(), cmap='gray')
    plt.show()
    return mean_loss


