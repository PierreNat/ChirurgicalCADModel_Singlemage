import tqdm
import numpy as np
import torch.nn as nn

def testResnet(model, test_dataloader, loss_function, file_name_extension, device, epoch_number=0):

    # test phase
    parameters = []  # ground truth labels
    predicted_params = []
    losses = []  # running loss

    count = 0
    steps_losses = []
    steps_alpha_loss = []
    steps_beta_loss = []
    steps_gamma_loss = []
    steps_x_loss = []
    steps_y_loss = []
    steps_z_loss = []

    loop = tqdm.tqdm(test_dataloader)
    for image, silhouette, parameter in loop:

        image = image.to(device)  # we have to send the inputs and targets at every step to the GPU too
        parameter = parameter.to(device)
        predicted_param = model(image)  # run prediction; output <- vector with probabilities of each class

        loss = loss_function(predicted_param, parameter) #MSE  value ?

        parameters.extend(parameter.detach().cpu().numpy())  # append ground truth parameters [array([...], dtype=float32), [...], dtype=float32),...)]
        predicted_params.extend(predicted_param.detach().cpu().numpy()) # append computed parameters

        alpha_loss = nn.MSELoss()(predicted_param[:, 0], parameter[:, 0])
        beta_loss = nn.MSELoss()(predicted_param[:, 1], parameter[:, 1])
        gamma_loss = nn.MSELoss()(predicted_param[:, 2], parameter[:, 2])
        x_loss = nn.MSELoss()(predicted_param[:, 3], parameter[:, 3])
        y_loss = nn.MSELoss()(predicted_param[:, 4], parameter[:, 4])
        z_loss = nn.MSELoss()(predicted_param[:, 5], parameter[:, 5])

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

    return parameters, predicted_params, this_epoch_loss, this_epoch_loss_alpha, this_epoch_loss_beta, this_epoch_loss_gamma, this_epoch_loss_x, this_epoch_loss_y, this_epoch_loss_z

