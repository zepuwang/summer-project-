from __future__ import division
import torch
from torch.autograd import Variable
import numpy as np
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
    calculate_picp,
    mean_prediction_interval_width,
    print_errors,
    mape,
    wape,
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


num_timesteps_output = 4
num_timesteps_input = num_timesteps_output

distribution_name = "M_Gaussian"
num_mixture = 2  # adjust it
best_model = "pth/model_Bike_DC_60min" + distribution_name + str(num_mixture) + ".pth"
print(best_model)
STmodel = torch.load(best_model).to(device=device)
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
# n_gaussian = 2
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
distribution_name = "M_Gaussian"
dis_hid = 16

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
)

epochs = 1  # 500

STmodel.load_state_dict(torch.load(best_model, map_location="cpu").state_dict())

X = X.astype(np.float32)
# X = X.reshape((X.shape[0],1,X.shape[1]))
split_line1 = int(X.shape[0] * 0.6)
split_line2 = int(X.shape[0] * 0.7)
print(X.shape, A.shape)


train_original_data = X[:split_line1, :]
val_original_data = X[split_line1:split_line2, :]
test_original_data = X[split_line2:, :]
print(train_original_data.shape)
print(val_original_data.shape)
print(test_original_data.shape)

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

A = torch.from_numpy(A).to(device="cpu")
print(test_target.shape)
STmodel.eval()
with torch.no_grad():

    test_input = test_input.to(device="cpu")
    test_target = test_target.to(device="cpu")

    test_loss_all = []
    test_pred_all = np.zeros_like(test_target)
    test_pred_all_var = np.zeros_like(test_target)

    print(test_input.shape, test_target.shape)
    for i in range(0, test_input.shape[0], batch_size):
        x_batch = test_input[i : i + batch_size, :, :]
        y_batch = test_target[i : i + batch_size, :, :]

        pi, mu, sigma = STmodel(x_batch, A)
        # mu = torch.round(mu)
        mu_1 = torch.sum(pi * mu, dim=-1)

        var_1 = (
            torch.sum(pi * sigma * sigma + pi * mu * mu, dim=-1)
            - torch.sum(pi * mu, dim=-1) ** 2
        )

        test_loss = STmodel.distribution.loss(y_batch, pi, mu, sigma)

        test_loss = np.asscalar(test_loss.detach().numpy())

        mu_1[mu_1 < 0] = 0

        mean_pred = mu_1
        test_pred_all[i : i + batch_size] = mean_pred
        test_pred_all_var[i : i + batch_size] = var_1

        test_loss_all.append(test_loss)

    pred = np.array(test_pred_all, dtype=np.float32)
    true = np.array(test_target, dtype=np.float32)
    var = np.array(test_pred_all_var, dtype=np.float32)
    print("without round")
    print_errors(true, pred, string=None)
    print("MAPE:")
    print(mape(true, pred))
    print("WAPE:")
    print(wape(true, pred))
    print("MPIW:")
    print(mean_prediction_interval_width(pred, var))
    print("PICP")
    print(calculate_picp(true, pred, var))
    print("with round")
    print_errors(true, np.round(pred), string=None)
    print("MAPE:")
    print(mape(true, pred))
    print("WAPE:")
    print(wape(true, pred))
    print("MPIW:")
    print(mean_prediction_interval_width(np.round(pred), var))
    print("PICP")
    print(calculate_picp(true, np.round(pred), var))
