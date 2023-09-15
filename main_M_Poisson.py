from enum import Enum, auto
import numpy as np
import matplotlib.pyplot as plt
from P_model import main_model, probability_model
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import torch.optim as optim
from torch import nn
import matplotlib.pyplot as plt
from utils import (
    generate_dataset,
    get_normalized_adj,
    get_Laplace,
    calculate_random_walk_matrix,
    nb_zeroinflated_nll_loss,
    nb_zeroinflated_draw,
    nb_MDN_nll_loss,
)
from model import *
import random, os, copy
import math
import tqdm
from scipy.stats import nbinom
import pickle as pk
import os
from P_model import probability_model


torch.manual_seed(0)
np.random.seed(42)

if torch.cuda.is_available():
    print("GPU is available.")
    print("Number of available GPUs:", torch.cuda.device_count())
    print(
        "GPU name:", torch.cuda.get_device_name(0)
    )  # Replace 0 with the GPU index if multiple GPUs are available.

    # Set the device to GPU
    device = torch.device(
        "cuda:0"
    )  # Replace 0 with the GPU index if multiple GPUs are available.

else:
    print("GPU is not available.")
    device = torch.device("cpu")

A = np.load("data/adj_Bike_DC_60.npy")  # change the loading folder
X = np.load("data/Bike_DC_60.npy")

"""
Model input and output
"""
num_timesteps_output = 4
num_timesteps_input = num_timesteps_output


"""
Model Hyperparameter
"""
space_dim = X.shape[1]
batch_size = 1
time_length = num_timesteps_output
nhid_spatial = 512
dropout = 0.2
alpha = 0.2
nheads_spatial = 4
temporal_hidden_dim = 32
num_heads_temporal = 4
fusion_d = 128
pred_len = num_timesteps_output
channels = [2, 4]
distribution_name = "M_Poisson"
dis_hid = 16
num_mixture = 2

"""
Set up model
"""

STmodel = probability_model(
    time_length,
    nhid_spatial,
    dropout,
    alpha,
    nheads_spatial,
    temporal_hidden_dim,
    pred_len,
    channels,
    dis_hid,
    num_mixture,
    distribution_name,
).to(device=device)

epochs = 500


X = X.astype(np.float32)
#X = X[:100,:]
split_line1 = int(X.shape[0] * 0.6)
split_line2 = int(X.shape[0] * 0.7)
print(X.shape, A.shape)


train_original_data = X[:split_line1, :]
val_original_data = X[split_line1:split_line2, :]
test_original_data = X[split_line2:, :]
print(train_original_data.shape)
training_input, training_target = generate_dataset(
    train_original_data,
    num_timesteps_input=num_timesteps_input,
    num_timesteps_output=num_timesteps_output,
)
val_input, val_target = generate_dataset(
    val_original_data,
    num_timesteps_input=num_timesteps_input,
    num_timesteps_output=num_timesteps_output,
)
test_input, test_target = generate_dataset(
    test_original_data,
    num_timesteps_input=num_timesteps_input,
    num_timesteps_output=num_timesteps_output,
)
print("input shape: ", training_input.shape, val_input.shape, test_input.shape)


A = torch.from_numpy(A).to(device=device)

optimizer = optim.Adam(STmodel.parameters(), lr=1e-4)
training_nll = []
validation_nll = []
validation_mae = []

for epoch in range(epochs):
    ## Step 1, training
    """
    # Begin training, similar training procedure from STGCN
    Trains one epoch with the given data.
    :param training_input: Training inputs of shape (num_samples, num_nodes,
    num_timesteps_train, num_features).
    :param training_target: Training targets of shape (num_samples, num_nodes,
    num_timesteps_predict).
    :param batch_size: Batch size to use during training.
    """
    epoch_training_losses = []
    total_loss = 0
    criterion = nn.MSELoss()
    for i in range(0, training_input.shape[0], batch_size):
        STmodel.train()
        optimizer.zero_grad()

        X_batch, y_batch = (
            training_input[i : i + batch_size, :, :],
            training_target[i : i + batch_size, :, :],
        )
        X_batch = X_batch.to(device=device)
        y_batch = y_batch.to(device=device)

        prob_success, pi = STmodel(X_batch, A)

        loss = STmodel.distribution.loss(y_batch, prob_success, pi) + criterion(
            y_batch, torch.sum(pi * prob_success, -1)
        )
        loss.backward()
        optimizer.step()
        total_loss += loss
        epoch_training_losses.append(loss.detach().cpu().numpy())
    training_nll.append(sum(epoch_training_losses) / len(epoch_training_losses))

    sub_validation_nll = []
    sub_validation_mae = []
    ## Step 2, validation
    for i in range(0, val_input.shape[0], batch_size):
        with torch.no_grad():
            STmodel.eval()
            X_batch, y_batch = (
                val_input[i : i + batch_size, :, :],
                val_target[i : i + batch_size, :, :],
            )
            X_batch = X_batch.to(device=device)
            y_batch = y_batch.to(device=device)

            prob_success, pi = STmodel(X_batch, A)
            # prob_success = prob_success.squeeze(-1)

            val_loss = STmodel.distribution.loss(y_batch, prob_success, pi) + criterion(
                y_batch, torch.sum(pi * prob_success, -1)
            )
            sub_validation_nll.append(val_loss.detach().cpu().numpy())

            val_pred = torch.sum(pi * prob_success, -1)
            mae = torch.mean(torch.abs(val_pred - y_batch))
            sub_validation_mae.append(mae.detach().cpu().numpy())
    val_loss = np.mean(np.array(sub_validation_nll))
    val_mae = np.mean(np.array(sub_validation_mae))
    validation_nll.append(val_loss)
    validation_mae.append(val_mae)

    ## Step 3, save the model
    """
    Save the model
    """

    print("Epoch: {}".format(epoch))
    print("Training loss: {}".format(training_nll[-1]))
    print(
        "Epoch %d: trainNLL %.5f; valNLL %.5f; mae %.4f"
        % (epoch, training_nll[-1], validation_nll[-1], validation_mae[-1])
    )
    if training_nll[-1] == min(training_nll):
        best_model = copy.deepcopy(STmodel.state_dict())
    checkpoint_path = "checkpoints/"
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    with open("checkpoints/losses.pk", "wb") as fd:
        pk.dump((training_nll, validation_nll, validation_mae), fd)
    if np.isnan(training_nll[-1]):
        break
STmodel.load_state_dict(best_model)
name = "pth/model_Bike_DC_60min" + distribution_name + str(num_mixture) + ".pth"
torch.save(STmodel, name)
