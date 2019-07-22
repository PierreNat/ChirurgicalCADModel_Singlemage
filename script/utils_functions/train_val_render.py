
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from utils_functions.renderBatchItem import renderBatchSil
from utils_functions.testRender import testRenderResnet

def train_render(model, train_dataloader, test_dataloader,
                 n_epochs, loss_function,
                 date4File, cubeSetName, batch_size, fileExtension, device, obj_name, noise):
    # monitor loss functions as the training progresses
    learning_rate = 0.001
    minRval = 0
    maxRval = 0
    minTXYval = -2
    maxTXYval = 2
    minTZval = 7
    maxTZval = 10
    all_Train_losses = []
    all_Test_losses = []

    Test_epoch_losses_alpha = []
    Test_epoch_losses_beta = []
    Test_epoch_losses_gamma = []
    Test_epoch_losses_x = []
    Test_epoch_losses_y = []
    Test_epoch_losses_z = []

    plot = False #plot the running renderered batch of image

    #file creation to store final values
    #contains 1 value per epoch for global loss, alpha , beta, gamma ,x, y, z validation loss
    epochsValLoss = open("./results/epochsValLoss_{}_{}_batchsOf{}img_{:.1f}%noise_{}epochs_RenderRegr.txt".format(date4File, cubeSetName, str(batch_size),  noise*100,  str(n_epochs), fileExtension), "w+")
    # contains 1 value per epoch for global loss, alpha , beta, gamma ,x, y, z training loss
    epochsTrainLoss = open("./results/epochsTrainLoss_{}_{}_batchsOf{}img_{:.1f}%noise_{}epochs_RenderRegr.txt".format(date4File, cubeSetName, str(batch_size), noise*100, str(n_epochs), fileExtension), "w+")
    # contains n steps value for global loss, alpha , beta, gamma ,x, y, z training loss
    stepsTrainLoss = open("./results/stepsTrainLoss_{}_{}_batchsOf{}img_{:.1f}%noise_{}epochs_RenderRegr.txt".format(date4File, cubeSetName, str(batch_size), noise*100, str(n_epochs), fileExtension), "w+")

    loop = n_epochs
    for epoch in tqdm(range(n_epochs)):

        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9) #use SGD with lr update

        print('RenderRegression run epoch: {} with Lr {} and noise {}%'.format(epoch, learning_rate, str(noise*100)))

        ## Training phase
        model.train()

        parameters = []  # ground truth labels
        predict_params = []  # predicted labels
        steps_losses = []  # contains the loss between silhouette after each steps
        steps_losses_MSE =[] # contains the loss mse computed with the  computed parameter and ground truth parameter
        steps_alpha_loss = []
        steps_beta_loss = []
        steps_gamma_loss = []
        steps_x_loss = []
        steps_y_loss = []
        steps_z_loss = []

        # loop = tqdm(train_dataloader)
        count = 0
        print('train phase epoch {}'.format(epoch))
        for image, silhouette, parameter in train_dataloader:
            image = image.to(device)  # we have to send the inputs and targets at every step to the GPU too
            silhouette = silhouette.to(device)

            #add noise to ground truth parameter
            Gt_val = parameter.cpu().numpy()
            #angle
            Gt_val[:, 0] = np.random.uniform(Gt_val[:, 0]-Gt_val[:, 0]*noise, Gt_val[:, 0]+Gt_val[:, 0]*noise)
            Gt_val[:, 1] = np.random.uniform(Gt_val[:, 1]-Gt_val[:, 1]*noise, Gt_val[:, 1]+Gt_val[:, 1]*noise)
            Gt_val[:, 2] = np.random.uniform(Gt_val[:, 2]-Gt_val[:, 2]*noise, Gt_val[:, 2]+Gt_val[:, 2]*noise)
            #translation
            Gt_val[:, 3] = np.random.uniform(Gt_val[:, 3]-Gt_val[:, 3]*noise, Gt_val[:, 3]+Gt_val[:, 3]*noise)
            Gt_val[:, 4] = np.random.uniform(Gt_val[:, 4]-Gt_val[:, 4]*noise, Gt_val[:, 4]+Gt_val[:, 4]*noise)
            Gt_val[:, 5] = np.random.uniform(Gt_val[:, 5]-Gt_val[:, 5]*noise, Gt_val[:, 5]+Gt_val[:, 5]*noise)

            parameter = torch.from_numpy(Gt_val)
            # Gt_val[:, 0] = Gt_val[:, 0] + np.random.normal(Gt_val[:, 0], 1, 1)
            parameter = parameter.to(device)


            #image has size [batch_length, 3, 512, 512]
            #predicted_param is a tensor with torch.size[batch, 6]
            predicted_params = model(image)  # run prediction; output <- vector containing  the 6 transformation params

            # m = nn.Sigmoid() #smooth function
            # predicted_params_Tsigmoid = m(predicted_params[:, 3:6]) #sigmoid function only on the Translation parameter of the batch
            # predicted_params[:, 3:6] = predicted_params_Tsigmoid  #create new array only for x y z value
            # predicted_params[:, 3:5] =  predicted_params[:, 3:5].clone()*(maxTXYval-minTXYval) + minTXYval  # rescale value to be between min max translation value allowed in x-y axis
            # predicted_params[:, 5] = predicted_params[:, 5].clone()*(maxTZval-minTZval) + minTZval         # rescale value to be at between min max translation value allowed in z axis

            # hard reset to 0 value that we don-t want to train
            # zero_array = torch.zeros(4, 5)
            # zero_array = zero_array.to(device)
            # predicted_params = torch.cat((zero_array, predicted_params), 1)
            # np_params = predicted_params.detach().cpu().numpy() #ensor to numpy array, ERROR HERE, DOES NOT HAVE GRAD

            if epoch % 20 == 0:
                plot = True
            else:
                plot = False

            # # zero the parameter gradients
            optimizer.zero_grad()

            # object, predicted, ground truth, loss , cuda , and bool for printing logic
            loss_function2 = nn.BCELoss()
            loss = renderBatchSil(obj_name, predicted_params, parameter, loss_function, device, plot) # loss between silhouette, mse or bce loss

            #one value each for the step, compute mse loss for all parameters separately
            lossMSE = loss_function(predicted_params, parameter)  # loss computed with computed value and ground truth
            alpha_loss = nn.MSELoss()(predicted_params[:, 0], parameter[:, 0])
            beta_loss = nn.MSELoss()(predicted_params[:, 1], parameter[:, 1])
            gamma_loss = nn.MSELoss()(predicted_params[:, 2], parameter[:, 2])
            x_loss = nn.MSELoss()(predicted_params[:, 3], parameter[:, 3])
            y_loss = nn.MSELoss()(predicted_params[:, 4], parameter[:, 4])
            z_loss = nn.MSELoss()(predicted_params[:, 5], parameter[:, 5])

            loss.backward()  # multiple times accumulates the gradient (by addition) for each parameter
            optimizer.step()  # performs a parameter update based on the current gradient, SGD is used here

            parameters.extend(parameter.cpu().numpy())  # append ground truth label
            predict_params.extend(predicted_params.detach().cpu().numpy())  # append computed parameters

            steps_losses.append(loss.item())  # loss between silhouette, mse or bce loss
            steps_losses_MSE.append(lossMSE.item())  # loss computed with computed value and ground truth
            steps_alpha_loss.append(alpha_loss.item())
            steps_beta_loss.append(beta_loss.item())
            steps_gamma_loss.append(gamma_loss.item())
            steps_x_loss.append(x_loss.item())
            steps_y_loss.append(y_loss.item())
            steps_z_loss.append(z_loss.item())

            print(
                'step: {}/{} current MSE loss : {:.4f}, MSE angle loss: {:.4f} {:.4f} {:.4f} MSE translation loss: {:.4f} {:.4f} {:.4f} '
                .format(count, loop, lossMSE, alpha_loss, beta_loss, gamma_loss, x_loss, y_loss, z_loss))

            #  save current step value for each parameter
            stepsTrainLoss.write(
                'step: {}/{} current MSE loss : {:.4f}, MSE angle loss: {:.4f} {:.4f} {:.4f}  MSE translation loss: {:.4f} {:.4f} {:.4f}  \r\n'
                .format(count, loop, lossMSE, alpha_loss, beta_loss, gamma_loss, x_loss, y_loss, z_loss))

            count = count + 1

        this_epoch_loss = np.mean(np.array(steps_losses))
        this_epoch_loss_MSE = np.mean(np.array(steps_losses_MSE))
        this_epoch_loss_alpha = np.mean(np.array(steps_alpha_loss))
        this_epoch_loss_beta = np.mean(np.array(steps_beta_loss))
        this_epoch_loss_gamma = np.mean(np.array(steps_gamma_loss))
        this_epoch_loss_x = np.mean(np.array(steps_x_loss))
        this_epoch_loss_y = np.mean(np.array(steps_y_loss))
        this_epoch_loss_z = np.mean(np.array(steps_z_loss))

        all_Train_losses.append(this_epoch_loss)  # will contain 1 loss per epoch

        epochsTrainLoss.write(
            'MSE loss for epoch {} is: {:.4f} MSE angle loss: {:.4f} {:.4f} {:.4f} MSE translation loss: {:.4f} {:.4f} {:.4f}  \r\n'
            .format(epoch, this_epoch_loss_MSE, this_epoch_loss_alpha, this_epoch_loss_beta, this_epoch_loss_gamma,
                    this_epoch_loss_x, this_epoch_loss_y, this_epoch_loss_z))


        torch.save(model.state_dict(),
                   './models/{}_TempModel_train_{}_batchsOf{}img_{:.1f}%noise_epochs_n{}_{}_RenderRegr.pth'.format(date4File, cubeSetName,
                                                                                            str(batch_size), noise*100, str(epoch),
                                                                                            fileExtension))

        print('parameters saved for epoch {}'.format(epoch))

        # # test the model
        # print('test phase epoch {}'.format(epoch))
        # model.eval()
        # parameters, predicted_params, test_losses, al, bl, gl, xl, yl, zl = testRenderResnet(model, test_dataloader, loss_function,
        #                                                                     fileExtension, device, obj_name,
        #                                                                     epoch_number=epoch)
        #
        # all_Test_losses.append(test_losses)
        # Test_epoch_losses_alpha.append(al)
        # Test_epoch_losses_beta.append(bl)
        # Test_epoch_losses_gamma.append(gl)
        # Test_epoch_losses_x.append(xl)
        # Test_epoch_losses_y.append(yl)
        # Test_epoch_losses_z.append(zl)
        #
        # epochsValLoss.write(
        #     'Validation Loss for epoch {} global {:.4f} angle loss: {:.4f} {:.4f} {:.4f} translation loss: {:.4f} {:.4f} {:.4f}  \r\n'
        #     .format(epoch, test_losses, al, bl, gl, xl, yl, zl))

    epochsValLoss.close()
    epochsTrainLoss.close()
    stepsTrainLoss.close()

    return all_Train_losses
