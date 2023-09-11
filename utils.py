from __future__ import division
import os
import zipfile
import numpy as np
import scipy.sparse as sp
import pandas as pd
from math import radians, cos, sin, asin, sqrt

# from sklearn.externals import joblib
import joblib
import scipy.io
import torch
from torch import nn
from scipy.stats import nbinom, norm

rand = np.random.RandomState(0)
"""
Geographical information calculation
"""


def get_long_lat(sensor_index, loc=None):
    """
    Input the index out from 0-206 to access the longitude and latitude of the nodes
    """
    if loc is None:
        locations = pd.read_csv("data/metr/graph_sensor_locations.csv")
    else:
        locations = loc
    lng = locations["longitude"].loc[sensor_index]
    lat = locations["latitude"].loc[sensor_index]
    return lng.to_numpy(), lat.to_numpy()


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371
    return c * r * 1000


"""
Generate the training sample for forecasting task, same idea from STGCN
"""


def generate_dataset(X, num_timesteps_input, num_timesteps_output):
    n = X.shape[0]
    features = []
    targets = []
    for i in range(n - num_timesteps_input - num_timesteps_output):
        feature = X[i : i + num_timesteps_input, :]
        target = X[
            i + num_timesteps_input : i + num_timesteps_input + num_timesteps_output, :
        ]
        features.append(feature)
        targets.append(target)

    return torch.from_numpy(np.array(features)).permute(0, 2, 1), torch.from_numpy(
        np.array(targets)
    ).permute(0, 2, 1)


"""
Dynamically construct the adjacent matrix
"""


def get_Laplace(A):
    """
    Returns the laplacian adjacency matrix. This is for C_GCN
    """
    if A[0, 0] == 1:
        A = A - np.diag(
            np.ones(A.shape[0], dtype=np.float32)
        )  # if the diag has been added by 1s
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A), diag.reshape((1, -1)))
    return A_wave


def get_normalized_adj(A):
    """
    Returns the degree normalized adjacency matrix. This is for K_GCN
    """
    if A[0, 0] == 0:
        A = A + np.diag(
            np.ones(A.shape[0], dtype=np.float32)
        )  # if the diag has been added by 1s
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5  # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A), diag.reshape((1, -1)))
    return A_wave


def calculate_random_walk_matrix(adj_mx):
    """
    Returns the random walk adjacency matrix. This is for D_GCN
    """
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.0
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx.toarray()


def test_error_virtual(STmodel, unknow_set, test_data, A_s, E_maxvalue, Missing0):
    """
    :param STmodel: The graph neural networks
    :unknow_set: The unknow locations for spatial prediction
    :test_data: The true value test_data of shape (test_num_timesteps, num_nodes)
    :A_s: The full adjacent matrix
    :Missing0: True: 0 in original datasets means missing data
    :return: NAE, MAPE and RMSE
    """
    unknow_set = set(unknow_set)
    time_dim = STmodel.time_dimension

    test_omask = np.ones(test_data.shape)
    if Missing0 == True:
        test_omask[test_data == 0] = 0
    test_inputs = (test_data * test_omask).astype("float32")
    test_inputs_s = test_inputs

    missing_index = np.ones(np.shape(test_data))
    missing_index[:, list(unknow_set)] = 0
    missing_index_s = missing_index

    o = np.zeros(
        [test_data.shape[0] // time_dim * time_dim, test_inputs_s.shape[1]]
    )  # Separate the test data into several h period

    for i in range(0, test_data.shape[0] // time_dim * time_dim, time_dim):
        inputs = test_inputs_s[i : i + time_dim, :]
        missing_inputs = missing_index_s[i : i + time_dim, :]
        T_inputs = inputs * missing_inputs
        T_inputs = T_inputs / E_maxvalue
        T_inputs = np.expand_dims(T_inputs, axis=0)
        T_inputs = torch.from_numpy(T_inputs.astype("float32"))
        A_q = torch.from_numpy((calculate_random_walk_matrix(A_s).T).astype("float32"))
        A_h = torch.from_numpy(
            (calculate_random_walk_matrix(A_s.T).T).astype("float32")
        )

        imputation = STmodel(T_inputs, A_q, A_h)
        imputation = imputation.data.numpy()
        o[i : i + time_dim, :] = imputation[0, :, :]

    o = o * E_maxvalue
    truth = test_inputs_s[0 : test_data.shape[0] // time_dim * time_dim]
    o[missing_index_s[0 : test_data.shape[0] // time_dim * time_dim] == 1] = truth[
        missing_index_s[0 : test_data.shape[0] // time_dim * time_dim] == 1
    ]

    test_mask = 1 - missing_index_s[0 : test_data.shape[0] // time_dim * time_dim]
    if Missing0 == True:
        test_mask[truth == 0] = 0
        o[truth == 0] = 0

    o_ = o[:, list(unknow_set)]
    truth_ = truth[:, list(unknow_set)]
    test_mask_ = test_mask[:, list(unknow_set)]

    MAE = np.sum(np.abs(o_ - truth_)) / np.sum(test_mask_)
    RMSE = np.sqrt(np.sum((o_ - truth_) * (o_ - truth_)) / np.sum(test_mask_))
    # MAPE = np.sum(np.abs(o - truth)/(truth + 1e-5))/np.sum( test_mask)
    R2 = 1 - np.sum((o_ - truth_) * (o_ - truth_)) / np.sum(
        (truth_ - truth_.mean()) * (truth_ - truth_.mean())
    )
    print(truth_.mean())
    return MAE, RMSE, R2, o


def test_error(STmodel, unknow_set, test_data, A_s, E_maxvalue, Missing0):
    """
    :param STmodel: The graph neural networks
    :unknow_set: The unknow locations for spatial prediction
    :test_data: The true value test_data of shape (test_num_timesteps, num_nodes)
    :A_s: The full adjacent matrix
    :Missing0: True: 0 in original datasets means missing data
    :return: NAE, MAPE and RMSE
    """
    unknow_set = set(unknow_set)
    time_dim = STmodel.time_dimension

    test_omask = np.ones(test_data.shape)
    if Missing0 == True:
        test_omask[test_data == 0] = 0
    test_inputs = (test_data * test_omask).astype("float32")
    test_inputs_s = test_inputs

    missing_index = np.ones(np.shape(test_data))
    missing_index[:, list(unknow_set)] = 0
    missing_index_s = missing_index

    o = np.zeros(
        [test_data.shape[0] // time_dim * time_dim, test_inputs_s.shape[1]]
    )  # Separate the test data into several h period

    for i in range(0, test_data.shape[0] // time_dim * time_dim, time_dim):
        inputs = test_inputs_s[i : i + time_dim, :]
        missing_inputs = missing_index_s[i : i + time_dim, :]
        T_inputs = inputs * missing_inputs
        T_inputs = T_inputs / E_maxvalue
        T_inputs = np.expand_dims(T_inputs, axis=0)
        T_inputs = torch.from_numpy(T_inputs.astype("float32"))
        A_q = torch.from_numpy((calculate_random_walk_matrix(A_s).T).astype("float32"))
        A_h = torch.from_numpy(
            (calculate_random_walk_matrix(A_s.T).T).astype("float32")
        )

        imputation = STmodel(T_inputs, A_q, A_h)
        imputation = imputation.data.numpy()
        o[i : i + time_dim, :] = imputation[0, :, :]

    o = o * E_maxvalue
    truth = test_inputs_s[0 : test_data.shape[0] // time_dim * time_dim]
    o[missing_index_s[0 : test_data.shape[0] // time_dim * time_dim] == 1] = truth[
        missing_index_s[0 : test_data.shape[0] // time_dim * time_dim] == 1
    ]

    test_mask = 1 - missing_index_s[0 : test_data.shape[0] // time_dim * time_dim]
    if Missing0 == True:
        test_mask[truth == 0] = 0
        o[truth == 0] = 0

    o_ = o[:, list(unknow_set)]
    truth_ = truth[:, list(unknow_set)]
    test_mask_ = test_mask[:, list(unknow_set)]

    MAE = np.sum(np.abs(o_ - truth_)) / np.sum(test_mask_)
    RMSE = np.sqrt(np.sum((o_ - truth_) * (o_ - truth_)) / np.sum(test_mask_))
    # MAPE = np.sum(np.abs(o - truth)/(truth + 1e-5))/np.sum( test_mask)
    R2 = 1 - np.sum((o_ - truth_) * (o_ - truth_)) / np.sum(
        (truth_ - truth_.mean()) * (truth_ - truth_.mean())
    )
    print(truth_.mean())
    return MAE, RMSE, R2, o


def rolling_test_error(STmodel, unknow_set, test_data, A_s, E_maxvalue, Missing0):
    """
    :It only calculates the last time points' prediction error, and updates inputs each time point
    :param STmodel: The graph neural networks
    :unknow_set: The unknow locations for spatial prediction
    :test_data: The true value test_data of shape (test_num_timesteps, num_nodes)
    :A_s: The full adjacent matrix
    :Missing0: True: 0 in original datasets means missing data
    :return: NAE, MAPE and RMSE
    """

    unknow_set = set(unknow_set)
    time_dim = STmodel.time_dimension

    test_omask = np.ones(test_data.shape)
    if Missing0 == True:
        test_omask[test_data == 0] = 0
    test_inputs = (test_data * test_omask).astype("float32")
    test_inputs_s = test_inputs

    missing_index = np.ones(np.shape(test_data))
    missing_index[:, list(unknow_set)] = 0
    missing_index_s = missing_index

    o = np.zeros([test_data.shape[0] - time_dim, test_inputs_s.shape[1]])

    for i in range(0, test_data.shape[0] - time_dim):
        inputs = test_inputs_s[i : i + time_dim, :]
        missing_inputs = missing_index_s[i : i + time_dim, :]
        MF_inputs = inputs * missing_inputs
        MF_inputs = np.expand_dims(MF_inputs, axis=0)
        MF_inputs = torch.from_numpy(MF_inputs.astype("float32"))
        A_q = torch.from_numpy((calculate_random_walk_matrix(A_s).T).astype("float32"))
        A_h = torch.from_numpy(
            (calculate_random_walk_matrix(A_s.T).T).astype("float32")
        )

        imputation = STmodel(MF_inputs, A_q, A_h)
        imputation = imputation.data.numpy()
        o[i, :] = imputation[0, time_dim - 1, :]

    truth = test_inputs_s[time_dim : test_data.shape[0]]
    o[missing_index_s[time_dim : test_data.shape[0]] == 1] = truth[
        missing_index_s[time_dim : test_data.shape[0]] == 1
    ]

    o = o * E_maxvalue
    truth = test_inputs_s[0 : test_data.shape[0] // time_dim * time_dim]
    test_mask = 1 - missing_index_s[time_dim : test_data.shape[0]]
    if Missing0 == True:
        test_mask[truth == 0] = 0
        o[truth == 0] = 0

    MAE = np.sum(np.abs(o - truth)) / np.sum(test_mask)
    RMSE = np.sqrt(np.sum((o - truth) * (o - truth)) / np.sum(test_mask))
    MAPE = np.sum(np.abs(o - truth) / (truth + 1e-5)) / np.sum(test_mask)  # avoid x/0

    return MAE, RMSE, MAPE, o


def test_error_cap(STmodel, unknow_set, full_set, test_set, A, time_dim, capacities):
    unknow_set = set(unknow_set)

    test_omask = np.ones(test_set.shape)
    test_omask[test_set == 0] = 0
    test_inputs = (test_set * test_omask).astype("float32")
    test_inputs_s = test_inputs  # [:, list(proc_set)]

    missing_index = np.ones(np.shape(test_inputs))
    missing_index[:, list(unknow_set)] = 0
    missing_index_s = missing_index  # [:, list(proc_set)]

    A_s = A  # [:, list(proc_set)][list(proc_set), :]
    o = np.zeros([test_set.shape[0] // time_dim * time_dim, test_inputs_s.shape[1]])

    for i in range(0, test_set.shape[0] // time_dim * time_dim, time_dim):
        inputs = test_inputs_s[i : i + time_dim, :]
        missing_inputs = missing_index_s[i : i + time_dim, :]
        MF_inputs = inputs * missing_inputs
        MF_inputs = MF_inputs
        MF_inputs = np.expand_dims(MF_inputs, axis=0)
        MF_inputs = torch.from_numpy(MF_inputs.astype("float32"))
        A_q = torch.from_numpy((calculate_random_walk_matrix(A_s).T).astype("float32"))
        A_h = torch.from_numpy(
            (calculate_random_walk_matrix(A_s.T).T).astype("float32")
        )

        imputation = STmodel(MF_inputs, A_q, A_h)
        imputation = imputation.data.numpy()
        o[i : i + time_dim, :] = imputation[0, :, :]

    o = o * capacities
    truth = test_inputs_s[0 : test_set.shape[0] // time_dim * time_dim]
    truth = truth * capacities
    o[missing_index_s[0 : test_set.shape[0] // time_dim * time_dim] == 1] = truth[
        missing_index_s[0 : test_set.shape[0] // time_dim * time_dim] == 1
    ]
    o[truth == 0] = 0

    test_mask = 1 - missing_index_s[0 : test_set.shape[0] // time_dim * time_dim]
    test_mask[truth == 0] = 0

    o_ = o[:, list(unknow_set)]
    truth_ = truth[:, list(unknow_set)]
    test_mask_ = test_mask[:, list(unknow_set)]

    MAE = np.sum(np.abs(o_ - truth_)) / np.sum(test_mask_)
    RMSE = np.sqrt(np.sum((o_ - truth_) * (o_ - truth_)) / np.sum(test_mask_))
    # MAPE = np.sum(np.abs(o - truth)/(truth + 1e-5))/np.sum( test_mask)
    R2 = 1 - np.sum((o_ - truth_) * (o_ - truth_)) / np.sum(
        (truth_ - truth_.mean()) * (truth_ - truth_.mean())
    )
    print(truth_.mean())
    return MAE, RMSE, R2, o


def nb_nll_loss(y, n, p, y_mask=None):
    """
    y: true values
    y_mask: whether missing mask is given
    """
    nll = (
        torch.lgamma(n)
        + torch.lgamma(y + 1)
        - torch.lgamma(n + y)
        - n * torch.log(p)
        - y * torch.log(1 - p)
    )
    if y_mask is not None:
        nll = nll * y_mask
    return torch.sum(nll)


def nb_zeroinflated_nll_loss(y, n, p, pi, y_mask=None):
    """
    y: true values
    y_mask: whether missing mask is given
    https://stats.idre.ucla.edu/r/dae/zinb/
    """
    idx_yeq0 = y == 0
    idx_yg0 = y > 0

    n_yeq0 = n[idx_yeq0]
    p_yeq0 = p[idx_yeq0]
    pi_yeq0 = pi[idx_yeq0]
    yeq0 = y[idx_yeq0]

    n_yg0 = n[idx_yg0]
    p_yg0 = p[idx_yg0]
    pi_yg0 = pi[idx_yg0]
    yg0 = y[idx_yg0]

    # L_yeq0 = torch.log(pi_yeq0) + (1-pi_yeq0)*torch.pow(p_yeq0,n_yeq0)
    # L_yg0  = torch.log(pi_yg0) + torch.lgamma(n_yg0+yg0) - torch.lgamma(yg0+1) - torch.lgamma(n_yg0) + n_yg0*torch.log(p_yg0) + yg0*torch.log(1-p_yg0)
    L_yeq0 = torch.log(pi_yeq0) + torch.log((1 - pi_yeq0) * torch.pow(p_yeq0, n_yeq0))
    L_yg0 = (
        torch.log(1 - pi_yg0)
        + torch.lgamma(n_yg0 + yg0)
        - torch.lgamma(yg0 + 1)
        - torch.lgamma(n_yg0)
        + n_yg0 * torch.log(p_yg0)
        + yg0 * torch.log(1 - p_yg0)
    )
    # print('nll',torch.mean(L_yeq0),torch.mean(L_yg0),torch.mean(torch.log(pi_yeq0)),torch.mean(torch.log(pi_yg0)))
    return -torch.sum(L_yeq0) - torch.sum(L_yg0)


def entropy_loss(y, pi):
    y_processed = torch.where(y != 0, torch.tensor(1), y)
    criterion = nn.BCELoss()
    loss = criterion(pi, y_processed).mean()
    return loss


import torch


def mdn_loss_fn(y, mu, sigma, pi):

    y += 1e-8

    y = torch.log(y)

    # Reshape y to match the shape of mu and sigma
    y = y.unsqueeze(-1).expand_as(mu)

    # Calculate the probability density function (PDF)
    sigma = sigma + 1e-8

    m = torch.distributions.Normal(loc=mu, scale=sigma)
    pdf = torch.exp(m.log_prob(y))

    # Calculate the weighted PDF and sum across mixture components

    weighted_pdf = pdf * pi  # * pi_yg0.unsqueeze(-1)
    # print(weighted_pdf[:,:10,:,:])
    sum_weighted_pdf = torch.sum(weighted_pdf, dim=-1)

    sum_weighted_pdf += 1e-8

    # Calculate the negative log-likelihood loss
    loss = -torch.log(sum_weighted_pdf)

    # Return the mean loss across the batch
    return torch.mean(loss)


""


def MDN_nll_loss(y, pi, pi_g, mu_g, sigma_g, y_mask=None):
    """
    y: true values
    y_mask: whether missing mask is given
    https://stats.idre.ucla.edu/r/dae/zinb/
    """
    idx_yeq0 = y == 0
    idx_yg0 = y > 0
    # pi_yq0 = pi[idx_yg0]
    pi_g_yg0 = idx_yg0.unsqueeze(-1) * pi_g
    mu_g_yg0 = mu_g * idx_yg0.unsqueeze(-1)
    sigma_yg0 = sigma_g * idx_yg0.unsqueeze(-1)

    sigma_g[sigma_g < 0] = 0

    loss = mdn_loss_fn(y, mu_g_yg0, sigma_yg0, pi_g_yg0)

    return loss


def nll_loss(y, pi, pi_g, mu_g, sigma_g, y_mask=None):
    idx_yeq0 = y == 0
    idx_yg0 = y > 0
    L1 = pi * idx_yeq0
    L2 = (1 - pi) * idx_yg0

    L1[L1 == 0] = 1e-8
    L2[L2 == 0] = 1e-8

    L1 = torch.log(L1)
    L2 = torch.log(L2)
    L = -L1 - L2
    return torch.mean(L)


def nb_MDN_nll_loss(y, pi, pi_g, mu_g, sigma_g, y_mask=None):
    return nll_loss(y, pi, pi_g, mu_g, sigma_g, y_mask=None) + MDN_nll_loss(
        y, pi, pi_g, mu_g, sigma_g, y_mask=None
    )


def nb_zeroinflated_draw(n, p, pi):
    """
    input: n, p, pi tensors
    output: drawn values
    """
    origin_shape = n.shape
    n = n.flatten()
    p = p.flatten()
    pi = pi.flatten()
    nb = nbinom(n, p)
    x_low = nb.ppf(0.01)
    x_up = nb.ppf(0.99)
    pred = np.zeros_like(n)
    # print(n.shape,x_low.shape,pi.min())
    for i in range(len(x_low)):
        if x_up[i] <= 1:
            x_up[i] = 1
        x = np.arange(x_low[i], x_up[i])
        # print(pi[0],pi[0].shape,x.shape,pi.shape)
        prob = (1 - pi[i]) * nbinom.pmf(x, n[i], p[i])
        #        print(len(prob),len(pi),len(n),len(x))
        prob[0] += pi[i]  # zero-inflatted
        pred[i] = rand.choice(
            a=x, p=prob / np.sum(prob)
        )  # random seed fixed, defined in the beginning

    return pred.reshape(origin_shape)


def gauss_draw(loc, scale):
    """
    input: n, p, pi tensors
    output: drawn values
    """
    origin_shape = loc.shape
    loc = loc.flatten()
    scale = scale.flatten()
    gauss = norm(loc, scale)
    x_low = gauss.ppf(0.01)
    x_up = gauss.ppf(0.99)
    pred = np.zeros_like(loc)
    # print(n.shape,x_low.shape,pi.min())
    for i in range(len(x_low)):
        x = np.arange(x_low[i], x_up[i], 100)
        prob = norm.pdf(x, loc[i], scale[i])
        pred[i] = rand.choice(
            a=x, p=prob / np.sum(prob)
        )  # random seed fixed, defined in the beginning

    return pred.reshape(origin_shape)


def nb_draw(n, p):
    """
    input: n, p, pi tensors
    output: drawn values
    """
    origin_shape = n.shape
    n = n.flatten()
    p = p.flatten()
    nb = nbinom(n, p)
    x_low = nb.ppf(0.01)
    x_up = nb.ppf(0.99)
    pred = np.zeros_like(n)
    for i in range(len(x_low)):
        if x_up[i] <= 1:
            x_up[i] = 1
        if x_up[i] == x_low[i]:
            x_up[i] = x_low[i] + 1
        # print(x_low[i],x_up[i])
        x = np.arange(x_low[i], x_up[i])
        prob = nbinom.pmf(x, n[i], p[i])
        pred[i] = rand.choice(
            a=x, p=prob / np.sum(prob)
        )  # random seed fixed, defined in the beginning

    return pred.reshape(origin_shape)


def gauss_loss(y, loc, scale, y_mask=None):
    """
    The location (loc) keyword specifies the mean. The scale (scale) keyword specifies the standard deviation.
    http://jrmeyer.github.io/machinelearning/2017/08/18/mle.html
    """
    torch.pi = (
        torch.acos(torch.zeros(1)).item() * 2
    )  # ugly define pi value in torch format
    LL = -1 / 2 * torch.log(2 * torch.pi * torch.pow(scale, 2)) - 1 / 2 * (
        torch.pow(y - loc, 2) / torch.pow(scale, 2)
    )
    return -torch.sum(LL)


def zero_inflated_mdn_loss(y, pi, pi_g, mu_g, sigma_g, y_mask=None):
    idx_yeq0 = y == 0
    idx_yg0 = y > 0

    pi_yeq0 = pi * idx_yeq0
    pi_yg0 = (1 - pi) * idx_yg0

    pi_g_yg0 = idx_yg0.unsqueeze(-1) * pi_g
    mu_g_yg0 = mu_g * idx_yg0.unsqueeze(-1)
    sigma_yg0 = sigma_g * idx_yg0.unsqueeze(-1)

    sigma_g[sigma_g < 0] = 0

    L_yeq0 = torch.log(pi_yeq0)
    L_yg0 = -torch.log(pi_yg0)  # + mdn_loss_fn(y,mu_g_yg0,sigma_yg0, pi_g_yg0,pi_yg0)

    return -torch.sum(L_yeq0) + torch.sum(L_yg0)


from scipy.stats import entropy
from scipy.special import kl_div


def rmse(truth, pred):
    return np.sqrt(((truth - pred) ** 2).mean())


def mae(truth, pred):
    return np.abs(truth - pred).mean()


def wape(truth, pred):
    return np.abs(np.subtract(pred, truth)).sum() / np.sum(truth)


def mape(truth, pred):
    return np.mean(np.abs((np.subtract(pred, truth) + 1e-5) / (truth + 1e-5)))


def true_zeros(truth, pred):
    idx = truth == 0
    return np.sum(pred[idx] == 0) / np.sum(idx)


def wrong_zeros(truth, pred):
    wrong_zeros = truth != 0
    pred_zeros = pred == 0
    # idx = wrong_zeros
    # array = truth

    precision = np.sum(pred_zeros & wrong_zeros) / np.sum(pred_zeros)
    return precision


def KL_DIV(truth, pred):
    return np.sum(pred * np.log((pred + 1e-7) / (truth + 1e-7)))


def KL_DIV_divide(truth, pred):
    return np.sum(pred * np.log((pred + 1e-7) / (truth + 1e-7))) / np.prod(truth.shape)


def F1_SCORE(truth, pred):
    true_zeros = truth == 0
    pred_zeros = pred == 0
    precision = np.sum(pred_zeros & true_zeros) / np.sum(pred_zeros)
    recall = np.sum(pred_zeros) / np.sum(true_zeros)
    return 2 * (precision * recall) / (precision + recall)


def mean_prediction_interval_width(mu, sigma, confidence=0.9):
    """
    Compute the Mean Prediction Interval Width (MPIW) on the specified confidence interval for a Gaussian distribution.

    Parameters:
    mu (numpy array): Array of mean values with shape (num_samples,).
    var (numpy array): Array of variance values with shape (num_samples,).
    confidence (float, optional): Confidence level for the prediction interval. Default is 0.9 (90%).

    Returns:
    float: Mean Prediction Interval Width (MPIW).
    """
    # Calculate the standard deviation from the variance
    std_dev = np.sqrt(sigma)

    # Compute the Z-score for the specified confidence level
    z_score = np.abs(
        np.percentile(
            np.random.normal(loc=0, scale=1, size=10000), (1 - confidence) / 2
        )
    )

    # Calculate the upper and lower bounds of the prediction intervals
    lower_bounds = mu - z_score * std_dev
    upper_bounds = mu + z_score * std_dev
    lower_bounds[lower_bounds < 0] = 0
    # Calculate the width of each prediction interval
    interval_widths = upper_bounds - lower_bounds

    # Compute the mean of the interval widths
    mpiw = np.mean(interval_widths)

    return mpiw


def calculate_picp(y_true, y_pred_mean, y_pred_variance, confidence=0.9):
    """
    Calculate Prediction Interval Coverage Probability (PICP).

    Parameters:
    - y_true: Actual observations (ground truth).
    - y_pred_mean: Predicted means from the NN.
    - y_pred_variance: Predicted variances from the NN.
    - alpha: Desired significance level. Default is 0.05 for 95% PI.

    Returns:
    - PICP value
    """
    # Calculate the z value from the standard normal distribution
    z = np.abs(
        np.percentile(np.random.standard_normal(100000), 100 * (1 - confidence) / 2)
    )

    # Calculate the prediction intervals
    lower_bound = y_pred_mean - z * np.sqrt(y_pred_variance)
    upper_bound = y_pred_mean + z * np.sqrt(y_pred_variance)

    # Calculate the number of true values that fall inside the prediction interval
    in_interval = np.sum((y_true >= lower_bound) & (y_true <= upper_bound))

    # Calculate PICP
    picp = in_interval / len(y_true.reshape(-1, 1))

    return picp


def print_errors(truth, pred, string=None):

    print(
        string,
        " RMSE %.4f MAE %.4f F1_SCORE %.4f KL-Div: %.4f, KL-Div-divide: %.4f, true_zeros_rate %.4f,wrong_zeros_rate %.4f : "
        % (
            rmse(truth, pred),
            mae(truth, pred),
            F1_SCORE(truth, pred),
            KL_DIV(truth, pred),
            KL_DIV_divide(truth, pred),
            true_zeros(truth, pred),
            wrong_zeros(truth, pred),
        ),
    )
