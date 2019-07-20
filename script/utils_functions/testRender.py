import tqdm
import numpy as np
import torch
import torch.nn as nn
from utils_functions.renderBatchItem import renderBatchSil

def testRenderResnet(model, test_dataloader, loss_function, file_name_extension, device , obj_name, epoch_number = 0):
    # monitor loss functions as the training progresses

    parameters = []  # ground truth labels
    predict_params = [] #computed labels

    loop = tqdm.tqdm(test_dataloader)
    count = 0
    steps_losses = []
    steps_alpha_loss = []
    steps_beta_loss = []
    steps_gamma_loss = []
    steps_x_loss = []
    steps_y_loss = []
    steps_z_loss = []
    # epochsTrainLoss = open("epochsTestLoss_epoch_{}_RenderRegr.txt".format(count), "w+")

    for image, silhouette, parameter in loop:
        image = image.to(device)  # we have to send the inputs and targets at every step to the GPU too
        silhouette = silhouette.to(device)
        parameter = parameter.to(device)

        #image has size [batch_length, 3, 512, 512]
        #predicted_param is a tensor with torch.siye[batch, 6]
        predicted_params = model(image)  # run prediction; output <- vector containing  the 6 transformation params


        # object, predicted, ground truth, loss , cuda , and bool for printing logic
        loss = renderBatchSil(obj_name, predicted_params, parameter, loss_function, device)

        parameters.extend(parameter.cpu().numpy())  # append ground truth label
        predict_params.extend(predicted_params.detach().cpu().numpy())  # append computed parameters

        alpha_loss = nn.MSELoss()(predicted_params[:, 0], parameter[:, 0])
        beta_loss = nn.MSELoss()(predicted_params[:, 1], parameter[:, 1])
        gamma_loss = nn.MSELoss()(predicted_params[:, 2], parameter[:, 2])
        x_loss = nn.MSELoss()(predicted_params[:, 3], parameter[:, 3])
        y_loss = nn.MSELoss()(predicted_params[:, 4], parameter[:, 4])
        z_loss = nn.MSELoss()(predicted_params[:, 5], parameter[:, 5])

        steps_losses.append(loss.item())  # only one loss value is add each step
        steps_alpha_loss.append(alpha_loss.item())
        steps_beta_loss.append(beta_loss.item())
        steps_gamma_loss.append(gamma_loss.item())
        steps_x_loss.append(x_loss.item())
        steps_y_loss.append(y_loss.item())
        steps_z_loss.append(z_loss.item())

        count = count + 1

    this_epoch_loss = np.mean(np.array(steps_losses))
    this_epoch_loss_alpha = np.mean(np.array(steps_alpha_loss))
    this_epoch_loss_beta = np.mean(np.array(steps_beta_loss))
    this_epoch_loss_gamma = np.mean(np.array(steps_gamma_loss))
    this_epoch_loss_x = np.mean(np.array(steps_x_loss))
    this_epoch_loss_y = np.mean(np.array(steps_y_loss))
    this_epoch_loss_z = np.mean(np.array(steps_z_loss))

    return parameters, predict_params, this_epoch_loss, this_epoch_loss_alpha, this_epoch_loss_beta, this_epoch_loss_gamma, this_epoch_loss_x, this_epoch_loss_y, this_epoch_loss_z

