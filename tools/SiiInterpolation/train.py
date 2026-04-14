"""
Based on
https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial2/Introduction_to_JAX.html
and the FLAX examples.

Script to train a NN on the training data.
The only input needed is to set the file path to the .h5 file containing the training data
The trained NN's are safed in the directory /tmp/checkpoints/
"""

import json
import time
import tqdm
from flax import nnx
import jax.numpy as jnp
import optax
import torch.utils.data as data
import h5py
import numpy as onp
import matplotlib.pyplot as plt
import orbax.checkpoint as ocp
import jax
import os
import re
from jaxrts.experimental.SiiNN import NNModel, NNModelExpandedZ


def l2_loss(x, alpha):
    return alpha * (x**2).sum()


def non_bias(path: tuple, value):
    return path[-1] != "bias"


non_bias_params = nnx.All(non_bias, nnx.Param)


def loss_fn(model, x, y):
    logits = model(x)
    mse = ((logits - y) ** 2).mean()

    # Add L2 weights regularization, see
    # https://github.com/google/flax/discussions/4160#discussioncomment-10556142
    # This was discussed by Dornheim 2019, for smoother solutions
    norm = sum(
        l2_loss(w, alpha=0)
        for w in jax.tree.leaves(nnx.state(model, non_bias_params))
    )
    loss = mse + norm
    return loss, mse


@nnx.jit
def train_step(
    model, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, x, y
):
    """Train for a single step."""
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, mse), grads = grad_fn(model, x, y)
    metrics.update(loss=loss, mse=mse)  # In-place updates.
    optimizer.update(grads)  # In-place updates.


@nnx.jit
def eval_step(model, metrics: nnx.MultiMetric, x, y):
    loss, mse = loss_fn(model, x, y)
    metrics.update(loss=loss, mse=mse)  # In-place updates.


def parse_data_path(path):
    # filename without directory or extension
    name = os.path.splitext(os.path.basename(path))[0]

    # expanded flag
    expanded_ionization = "expanded" in name

    # split formula and datapoints
    match = re.match(r"([A-Za-z0-9\.]+)_(\d+)", name)
    formula = match.group(1)
    datapoints = int(match.group(2))

    # extract elements after numeric ratios
    elements = re.findall(r"\d+\.\d+([A-Z][a-z]?)", formula)

    return expanded_ionization, elements, datapoints


class Dataset1C(data.Dataset):
    def __init__(self, hdf5_file_path, spec1):
        with h5py.File(hdf5_file_path, "r") as hdf5_file:
            self.T = hdf5_file["inputs"]["T"][:]
            self.rho = hdf5_file["inputs"]["rho"][:]
            self.Z1 = hdf5_file["inputs"][f"Z_{spec1}"][:]
            self.k = hdf5_file["inputs"]["k"][:]
            self.S_11 = hdf5_file["outputs"][f"S_{spec1}{spec1}"][:]

        self.elements = [spec1]

        mask = ~jnp.isnan(self.S_11)

        self.T = self.T[mask]
        self.rho = self.rho[mask]
        self.Z1 = self.Z1[mask]
        self.k = self.k[mask]
        self.S_11 = self.S_11[mask]

        self.sf_T = jnp.max(self.T)
        self.sf_rho = jnp.max(self.rho)
        self.sf_Z1 = jnp.max(self.Z1)
        self.sf_k = jnp.max(self.k)

        self.s_T = self.T / self.sf_T
        self.s_rho = self.rho / self.sf_rho
        self.s_Z1 = self.Z1 / self.sf_Z1
        self.s_k = self.k / self.sf_k
        self.inputs = onp.array(
            [
                self.s_T,
                self.s_rho,
                self.s_Z1,
                self.s_k,
            ]
        )

        self.outputs = onp.array([self.S_11])

    def __len__(self):
        return len(self.S_11)

    def __getitem__(self, idx):
        return self.inputs[:, idx], self.outputs[:, idx]


class Dataset1C_expanded_ionization_state(data.Dataset):
    def __init__(self, hdf5_file_path, spec1, spec2):
        with h5py.File(hdf5_file_path, "r") as hdf5_file:
            self.T = hdf5_file["inputs"]["T"][:]
            self.rho = hdf5_file["inputs"]["rho"][:]
            self.Z1 = hdf5_file["inputs"][f"Z_{spec1}"][:]
            self.k = hdf5_file["inputs"]["k"][:]
            self.S_11 = hdf5_file["outputs"][f"S_{spec1}{spec1}"][:]
            try:
                self.S_12 = hdf5_file["outputs"][f"S_{spec1}{spec2}"][:]
            except KeyError:
                self.S_12 = hdf5_file["outputs"][f"S_{spec2}{spec1}"][:]

            self.S_22 = hdf5_file["outputs"][f"S_{spec2}{spec2}"][:]

        self.elements = [spec1, spec2]

        mask = (
            (~jnp.isnan(self.S_22))
            & (~jnp.isnan(self.S_12))
            & (~jnp.isnan(self.S_11))
        )

        self.T = self.T[mask]
        self.rho = self.rho[mask]
        self.Z1 = self.Z1[mask]
        self.k = self.k[mask]
        self.S_11 = self.S_11[mask]
        self.S_12 = self.S_12[mask]
        self.S_22 = self.S_22[mask]

        self.sf_T = jnp.max(self.T)
        self.sf_rho = jnp.max(self.rho)
        self.sf_Z1 = jnp.max(self.Z1)
        self.sf_k = jnp.max(self.k)

        self.s_T = self.T / self.sf_T
        self.s_rho = self.rho / self.sf_rho
        self.s_Z1 = self.Z1 / self.sf_Z1
        self.s_k = self.k / self.sf_k

        self.inputs = onp.array(
            [
                self.s_T,
                self.s_rho,
                self.s_Z1,
                self.s_k,
            ]
        )
        self.outputs = onp.array(
            [
                self.S_11,
                self.S_12,
                self.S_22,
            ]
        )

    def __len__(self):
        return len(self.S_11)

    def __getitem__(self, idx):

        return self.inputs[:, idx], self.outputs[:, idx]


class Dataset2C(data.Dataset):
    def __init__(self, hdf5_file_path, spec1, spec2):
        with h5py.File(hdf5_file_path, "r") as hdf5_file:
            self.T = hdf5_file["inputs"]["T"][:]
            self.rho = hdf5_file["inputs"]["rho"][:]
            self.Z1 = hdf5_file["inputs"][f"Z_{spec1}"][:]
            self.Z2 = hdf5_file["inputs"][f"Z_{spec2}"][:]
            self.k = hdf5_file["inputs"]["k"][:]
            self.S_11 = hdf5_file["outputs"][f"S_{spec1}{spec1}"][:]
            try:
                self.S_12 = hdf5_file["outputs"][f"S_{spec1}{spec2}"][:]
            except KeyError:
                self.S_12 = hdf5_file["outputs"][f"S_{spec2}{spec1}"][:]
            self.S_22 = hdf5_file["outputs"][f"S_{spec2}{spec2}"][:]

        self.elements = [spec1, spec2]

        mask = (
            (~jnp.isnan(self.S_22))
            & (~jnp.isnan(self.S_12))
            & (~jnp.isnan(self.S_11))
        )

        self.T = self.T[mask]
        self.rho = self.rho[mask]
        self.Z1 = self.Z1[mask]
        self.Z2 = self.Z2[mask]
        self.k = self.k[mask]
        self.S_11 = self.S_11[mask]
        self.S_12 = self.S_12[mask]
        self.S_22 = self.S_22[mask]

        self.sf_T = jnp.max(self.T)
        self.sf_rho = jnp.max(self.rho)
        self.sf_Z1 = jnp.max(self.Z1)
        self.sf_Z2 = jnp.max(self.Z2)
        self.sf_k = jnp.max(self.k)

        self.s_T = self.T / self.sf_T
        self.s_rho = self.rho / self.sf_rho
        self.s_Z2 = self.Z2 / self.sf_Z2
        self.s_Z1 = self.Z1 / self.sf_Z1
        self.s_k = self.k / self.sf_k

        self.inputs = onp.array(
            [
                self.s_T,
                self.s_rho,
                self.s_Z1,
                self.s_Z2,
                self.s_k,
            ]
        )
        self.outputs = onp.array(
            [
                self.S_11,
                self.S_12,
                self.S_22,
            ]
        )

    def __len__(self):
        return len(self.S_11)

    def __getitem__(self, idx):

        return self.inputs[:, idx], self.outputs[:, idx]


class Dataset2C_expanded_ionization_state(data.Dataset):
    def __init__(self, hdf5_file_path, spec1, spec2, spec3, spec4):
        with h5py.File(hdf5_file_path, "r") as hdf5_file:
            self.T = hdf5_file["inputs"]["T"][:]
            self.rho = hdf5_file["inputs"]["rho"][:]

            self.Z1 = hdf5_file["inputs"][f"Z_{spec1}"][:]
            self.Z2 = hdf5_file["inputs"][f"Z_{spec3}"][:]

            self.k = hdf5_file["inputs"]["k"][:]
            self.S_11 = hdf5_file["outputs"][f"S_{spec1}{spec1}"][:]
            self.S_22 = hdf5_file["outputs"][f"S_{spec2}{spec2}"][:]
            self.S_33 = hdf5_file["outputs"][f"S_{spec3}{spec3}"][:]
            self.S_44 = hdf5_file["outputs"][f"S_{spec4}{spec4}"][:]

            try:
                self.S_12 = hdf5_file["outputs"][f"S_{spec1}{spec2}"][:]
            except KeyError:
                self.S_12 = hdf5_file["outputs"][f"S_{spec2}{spec1}"][:]
            try:
                self.S_13 = hdf5_file["outputs"][f"S_{spec1}{spec3}"][:]
            except KeyError:
                self.S_13 = hdf5_file["outputs"][f"S_{spec3}{spec1}"][:]
            try:
                self.S_14 = hdf5_file["outputs"][f"S_{spec1}{spec4}"][:]
            except KeyError:
                self.S_14 = hdf5_file["outputs"][f"S_{spec4}{spec1}"][:]

            try:
                self.S_23 = hdf5_file["outputs"][f"S_{spec2}{spec3}"][:]
            except KeyError:
                self.S_23 = hdf5_file["outputs"][f"S_{spec3}{spec2}"][:]
            try:
                self.S_24 = hdf5_file["outputs"][f"S_{spec2}{spec4}"][:]
            except KeyError:
                self.S_24 = hdf5_file["outputs"][f"S_{spec4}{spec2}"][:]
            try:
                self.S_34 = hdf5_file["outputs"][f"S_{spec3}{spec4}"][:]
            except KeyError:
                self.S_34 = hdf5_file["outputs"][f"S_{spec4}{spec3}"][:]

        self.elements = [spec1, spec2, spec3, spec4]

        ## TODO
        mask = (
            (~jnp.isnan(self.S_11))
            & (~jnp.isnan(self.S_12))
            & (~jnp.isnan(self.S_13))
            & (~jnp.isnan(self.S_14))
            & (~jnp.isnan(self.S_22))
            & (~jnp.isnan(self.S_23))
            & (~jnp.isnan(self.S_24))
            & (~jnp.isnan(self.S_33))
            & (~jnp.isnan(self.S_34))
            & (~jnp.isnan(self.S_44))
        )

        self.T = self.T[mask]
        self.rho = self.rho[mask]
        self.Z1 = self.Z1[mask]
        self.Z2 = self.Z2[mask]
        self.k = self.k[mask]
        self.S_11 = self.S_11[mask]
        self.S_12 = self.S_12[mask]
        self.S_13 = self.S_13[mask]
        self.S_14 = self.S_14[mask]
        self.S_22 = self.S_22[mask]
        self.S_23 = self.S_23[mask]
        self.S_24 = self.S_24[mask]
        self.S_33 = self.S_33[mask]
        self.S_34 = self.S_34[mask]
        self.S_44 = self.S_44[mask]

        self.sf_T = jnp.max(self.T)
        self.sf_rho = jnp.max(self.rho)
        self.sf_Z1 = jnp.max(self.Z1)
        self.sf_Z2 = jnp.max(self.Z2)
        self.sf_k = jnp.max(self.k)

        self.s_T = self.T / self.sf_T
        self.s_rho = self.rho / self.sf_rho
        self.s_Z2 = self.Z2 / self.sf_Z2
        self.s_Z1 = self.Z1 / self.sf_Z1
        self.s_k = self.k / self.sf_k

        self.inputs = onp.array(
            [
                self.s_T,
                self.s_rho,
                self.s_Z1,
                self.s_Z2,
                self.s_k,
            ]
        )
        self.outputs = onp.array(
            [
                self.S_11,
                self.S_12,
                self.S_13,
                self.S_14,
                self.S_22,
                self.S_23,
                self.S_24,
                self.S_33,
                self.S_34,
                self.S_44,
            ]
        )

    def __len__(self):
        return len(self.S_11)

    def __getitem__(self, idx):

        return self.inputs[:, idx], self.outputs[:, idx]


class Dataset3C(data.Dataset):
    def __init__(self, hdf5_file_path, spec1, spec2, spec3):
        with h5py.File(hdf5_file_path, "r") as hdf5_file:
            self.T = hdf5_file["inputs"]["T"][:]
            self.rho = hdf5_file["inputs"]["rho"][:]

            self.Z1 = hdf5_file["inputs"][f"Z_{spec1}"][:]
            self.Z2 = hdf5_file["inputs"][f"Z_{spec2}"][:]
            self.Z3 = hdf5_file["inputs"][f"Z_{spec3}"][:]

            self.k = hdf5_file["inputs"]["k"][:]
            self.S_11 = hdf5_file["outputs"][f"S_{spec1}{spec1}"][:]
            self.S_22 = hdf5_file["outputs"][f"S_{spec2}{spec2}"][:]
            self.S_33 = hdf5_file["outputs"][f"S_{spec3}{spec3}"][:]

            try:
                self.S_12 = hdf5_file["outputs"][f"S_{spec1}{spec2}"][:]
            except KeyError:
                self.S_12 = hdf5_file["outputs"][f"S_{spec2}{spec1}"][:]
            try:
                self.S_13 = hdf5_file["outputs"][f"S_{spec1}{spec3}"][:]
            except KeyError:
                self.S_13 = hdf5_file["outputs"][f"S_{spec3}{spec1}"][:]

            try:
                self.S_23 = hdf5_file["outputs"][f"S_{spec2}{spec3}"][:]
            except KeyError:
                self.S_23 = hdf5_file["outputs"][f"S_{spec3}{spec2}"][:]

        self.elements = [spec1, spec2, spec3]

        ## TODO
        mask = (
            (~jnp.isnan(self.S_11))
            & (~jnp.isnan(self.S_12))
            & (~jnp.isnan(self.S_13))
            & (~jnp.isnan(self.S_22))
            & (~jnp.isnan(self.S_23))
            & (~jnp.isnan(self.S_33))
        )

        self.T = self.T[mask]
        self.rho = self.rho[mask]

        self.Z1 = self.Z1[mask]
        self.Z2 = self.Z2[mask]
        self.Z3 = self.Z3[mask]

        self.k = self.k[mask]
        self.S_11 = self.S_11[mask]
        self.S_12 = self.S_12[mask]
        self.S_13 = self.S_13[mask]
        self.S_22 = self.S_22[mask]
        self.S_23 = self.S_23[mask]
        self.S_33 = self.S_33[mask]

        self.sf_T = jnp.max(self.T)
        self.sf_rho = jnp.max(self.rho)
        self.sf_Z1 = jnp.max(self.Z1)
        self.sf_Z2 = jnp.max(self.Z2)
        self.sf_Z3 = jnp.max(self.Z3)
        self.sf_k = jnp.max(self.k)

        self.s_T = self.T / self.sf_T
        self.s_rho = self.rho / self.sf_rho
        self.s_Z1 = self.Z1 / self.sf_Z1
        self.s_Z2 = self.Z2 / self.sf_Z2
        self.s_Z3 = self.Z3 / self.sf_Z3
        self.s_k = self.k / self.sf_k

        self.inputs = onp.array(
            [
                self.s_T,
                self.s_rho,
                self.s_Z1,
                self.s_Z2,
                self.s_Z3,
                self.s_k,
            ]
        )
        self.outputs = onp.array(
            [
                self.S_11,
                self.S_12,
                self.S_13,
                self.S_22,
                self.S_23,
                self.S_33,
            ]
        )

    def __len__(self):
        return len(self.S_11)

    def __getitem__(self, idx):

        return self.inputs[:, idx], self.outputs[:, idx]


class Dataset4C(data.Dataset):
    def __init__(self, hdf5_file_path, spec1, spec2, spec3, spec4):
        with h5py.File(hdf5_file_path, "r") as hdf5_file:
            self.T = hdf5_file["inputs"]["T"][:]
            self.rho = hdf5_file["inputs"]["rho"][:]

            self.Z1 = hdf5_file["inputs"][f"Z_{spec1}"][:]
            self.Z2 = hdf5_file["inputs"][f"Z_{spec2}"][:]
            self.Z3 = hdf5_file["inputs"][f"Z_{spec3}"][:]
            self.Z4 = hdf5_file["inputs"][f"Z_{spec4}"][:]

            self.k = hdf5_file["inputs"]["k"][:]
            self.S_11 = hdf5_file["outputs"][f"S_{spec1}{spec1}"][:]
            self.S_22 = hdf5_file["outputs"][f"S_{spec2}{spec2}"][:]
            self.S_33 = hdf5_file["outputs"][f"S_{spec3}{spec3}"][:]
            self.S_44 = hdf5_file["outputs"][f"S_{spec4}{spec4}"][:]

            try:
                self.S_12 = hdf5_file["outputs"][f"S_{spec1}{spec2}"][:]
            except KeyError:
                self.S_12 = hdf5_file["outputs"][f"S_{spec2}{spec1}"][:]
            try:
                self.S_13 = hdf5_file["outputs"][f"S_{spec1}{spec3}"][:]
            except KeyError:
                self.S_13 = hdf5_file["outputs"][f"S_{spec3}{spec1}"][:]
            try:
                self.S_14 = hdf5_file["outputs"][f"S_{spec1}{spec4}"][:]
            except KeyError:
                self.S_14 = hdf5_file["outputs"][f"S_{spec4}{spec1}"][:]

            try:
                self.S_23 = hdf5_file["outputs"][f"S_{spec2}{spec3}"][:]
            except KeyError:
                self.S_23 = hdf5_file["outputs"][f"S_{spec3}{spec2}"][:]
            try:
                self.S_24 = hdf5_file["outputs"][f"S_{spec2}{spec4}"][:]
            except KeyError:
                self.S_24 = hdf5_file["outputs"][f"S_{spec4}{spec2}"][:]
            try:
                self.S_34 = hdf5_file["outputs"][f"S_{spec3}{spec4}"][:]
            except KeyError:
                self.S_34 = hdf5_file["outputs"][f"S_{spec4}{spec3}"][:]

        self.elements = [spec1, spec2, spec3, spec4]

        ## TODO
        mask = (
            (~jnp.isnan(self.S_11))
            & (~jnp.isnan(self.S_12))
            & (~jnp.isnan(self.S_13))
            & (~jnp.isnan(self.S_14))
            & (~jnp.isnan(self.S_22))
            & (~jnp.isnan(self.S_23))
            & (~jnp.isnan(self.S_24))
            & (~jnp.isnan(self.S_33))
            & (~jnp.isnan(self.S_34))
            & (~jnp.isnan(self.S_44))
        )

        self.T = self.T[mask]
        self.rho = self.rho[mask]

        self.Z1 = self.Z1[mask]
        self.Z2 = self.Z2[mask]
        self.Z3 = self.Z3[mask]
        self.Z4 = self.Z4[mask]

        self.k = self.k[mask]
        self.S_11 = self.S_11[mask]
        self.S_12 = self.S_12[mask]
        self.S_13 = self.S_13[mask]
        self.S_14 = self.S_14[mask]
        self.S_22 = self.S_22[mask]
        self.S_23 = self.S_23[mask]
        self.S_24 = self.S_24[mask]
        self.S_33 = self.S_33[mask]
        self.S_34 = self.S_34[mask]
        self.S_44 = self.S_44[mask]

        self.sf_T = jnp.max(self.T)
        self.sf_rho = jnp.max(self.rho)
        self.sf_Z1 = jnp.max(self.Z1)
        self.sf_Z2 = jnp.max(self.Z2)
        self.sf_Z3 = jnp.max(self.Z3)
        self.sf_Z4 = jnp.max(self.Z4)
        self.sf_k = jnp.max(self.k)

        self.s_T = self.T / self.sf_T
        self.s_rho = self.rho / self.sf_rho
        self.s_Z1 = self.Z1 / self.sf_Z1
        self.s_Z2 = self.Z2 / self.sf_Z2
        self.s_Z3 = self.Z3 / self.sf_Z3
        self.s_Z4 = self.Z4 / self.sf_Z4
        self.s_k = self.k / self.sf_k

        self.inputs = onp.array(
            [
                self.s_T,
                self.s_rho,
                self.s_Z1,
                self.s_Z2,
                self.s_Z3,
                self.s_Z4,
                self.s_k,
            ]
        )
        self.outputs = onp.array(
            [
                self.S_11,
                self.S_12,
                self.S_13,
                self.S_14,
                self.S_22,
                self.S_23,
                self.S_24,
                self.S_33,
                self.S_34,
                self.S_44,
            ]
        )

    def __len__(self):
        return len(self.S_11)

    def __getitem__(self, idx):

        return self.inputs[:, idx], self.outputs[:, idx]


# This collate function is taken from the JAX tutorial with PyTorch Data
# Loading
# https://jax.readthedocs.io/en/latest/notebooks/Neural_Network_and_Data_Loading.html
def numpy_collate(batch):
    if isinstance(batch[0], onp.ndarray):
        return onp.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return onp.array(batch)


##############################################################################
########################### TRAIN SCRIPT START ###############################
##############################################################################

## Set training data file on which the NN is trained
data_filepath = f"train_data/1.0C1.0H_200000_expanded.h5"

# Set NN training parameters
hidden_layers = [64, 128, 128, 64]
learning_rate = 1e-4
nr_epoches = 2000

# Determine needed information from file path
expanded_ionization, Elements, number_points = parse_data_path(
    path=data_filepath
)

# set a correctiong value when a plasma state is expanded
correction_value = 1
if expanded_ionization:
    correction_value = 2
    expanded_elements = [i for i in Elements for _ in range(2)]
    Elements = []
    for i, elem in enumerate(expanded_elements):
        if i % 2 == 1:
            Elements.append(elem + "+")
        else:
            Elements.append(elem)

## Select Dataset for the training data
nr_of_ions = len(Elements)
if nr_of_ions == 1:
    print("I use Dataset1C")
    dataset = Dataset1C(data_filepath, *Elements)

elif nr_of_ions == 2:
    if expanded_ionization:
        print("I use Dataset1C_expanded")
        dataset = Dataset1C_expanded_ionization_state(data_filepath, *Elements)
    else:
        print("I use Dataset2C")
        dataset = Dataset2C(data_filepath, *Elements)

elif nr_of_ions == 3:
    print("I use Dataset3C")
    dataset = Dataset3C(data_filepath, *Elements)

elif nr_of_ions == 4:
    if expanded_ionization:
        print("I use Dataset2C_expanded")
        dataset = Dataset2C_expanded_ionization_state(data_filepath, *Elements)
    else:
        print("I use Dataset4C")
        dataset = Dataset4C(data_filepath, *Elements)
else:
    raise ValueError(
        "Nr_of_ions > 4, for which no Dataset class is defined yet!"
    )

### Define model shape depending on how many Elements are considered in plasmastate
### NNModel(intput_layer: Nr_ions + 3, [hidden layers], output_layer: Nr_of_unique Sii's: n(n+1) / 2,
### where n is the number of ions for non-expanded, and twice the number of ions for an expanded PlasmaState
ApproriateModelClass = NNModelExpandedZ if expanded_ionization else NNModel
model = ApproriateModelClass(
    int(nr_of_ions // correction_value + 3),
    hidden_layers,
    nr_of_ions * (nr_of_ions + 1) // 2,
    nnx.Rngs(int(1e6 * time.time()) % 42),
)
## Select optimizer
optimizer = nnx.ModelAndOptimizer(
    model, optax.adam(learning_rate=learning_rate)
)

## Set norms for Z correctly according to how many elements are in your plasma state
## and whether the state is expanded

if nr_of_ions == 1 or (nr_of_ions == 2 and expanded_ionization == True):
    Z_norm = [float(dataset.sf_Z1)]

elif nr_of_ions == 2 or (nr_of_ions == 4 and expanded_ionization == True):
    Z_norm = [float(dataset.sf_Z1), float(dataset.sf_Z2)]

elif nr_of_ions == 3 or (nr_of_ions == 6 and expanded_ionization == True):
    Z_norm = [float(dataset.sf_Z1), float(dataset.sf_Z2), float(dataset.sf_Z3)]

elif nr_of_ions == 4 or (nr_of_ions == 8 and expanded_ionization == True):
    Z_norm = [
        float(dataset.sf_Z1),
        float(dataset.sf_Z2),
        float(dataset.sf_Z3),
        float(dataset.sf_Z4),
    ]
else:
    raise ValueError

# Set the norms, extracted from the dataset
model.set_norms(
    T=float(dataset.sf_T),
    rho=float(dataset.sf_rho),
    Z=Z_norm,
    k=float(dataset.sf_k),
)
print(model.norms)

train_dataset, test_dataset = data.random_split(dataset, [0.8, 0.2])

batch_size = 128
train_loader = data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=numpy_collate,
)
test_loader = data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=True, collate_fn=numpy_collate
)


# Test sizes of in- and outputs
train_inputs, train_outputs = next(iter(train_loader))
print("train inputs, shape:", train_inputs.shape)
print("train labels, shape:", train_outputs.shape)
print(f"Size traindataset {len(train_loader)}")
print(f"Size testdataset  {len(test_loader)}")

# Save metrics in an object
metrics = nnx.MultiMetric(
    loss=nnx.metrics.Average("loss"),
    accuracy=nnx.metrics.Average("mse"),
)

# Set checkpoint dir where the trained NN's are safed every 100 epoches
ckpt_dir = ocp.test_utils.erase_and_create_empty("/tmp/checkpoints/")
checkpointer = ocp.StandardCheckpointer()


def train_model(model, train_loader, test_loader, metrics, num_epochs=1001):
    """
    Function to train the NN --  perform forward and backward propagation and update
    weights and biases.
    """
    metrics_history = {
        "train_loss": [],
        "test_loss": [],
        "train_accuracy": [],
        "test_accuracy": [],
    }
    for epoch_num, epoch in tqdm.tqdm(
        enumerate(range(num_epochs)), total=num_epochs
    ):
        for batch in train_loader:
            train_step(model, optimizer, metrics, *batch)

        for metric, value in metrics.compute().items():  # Compute the metrics.
            metrics_history[f"train_{metric}"].append(value)
        metrics.reset()  # Reset the metrics for the test set.

        for batch in test_loader:
            eval_step(model, metrics, *batch)

        # Log the test metrics.
        for metric, value in metrics.compute().items():
            metrics_history[f"test_{metric}"].append(value)
        metrics.reset()  # Reset the metrics for the next training epoch.
        if (epoch_num > 0) & (epoch_num % 100 == 0):
            checkpoint_path = ckpt_dir / f"epoch_{epoch_num}/"
            (
                _,
                state,
            ) = nnx.split(model)

            checkpointer.save(
                checkpoint_path,
                state,
            )
            checkpointer.wait_until_finished()
            print(checkpoint_path / "SHAPE")
            with open(checkpoint_path / "SHAPE", "w") as f:
                shape = {
                    "din": model.din,
                    "dhid": model.dhid,
                    "dout": model.dout,
                    "no_of_atoms": len(model.norm_Z),
                    "elements": dataset.elements,
                    "expanded": expanded_ionization,
                }
                json.dump(shape, f)

    return metrics_history


metrics_hist = train_model(
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    metrics=metrics,
    num_epochs=nr_epoches + 1,
)

fig, ax = plt.subplots(1)
ax.plot(metrics_hist["train_loss"], alpha=0.8, label="train_loss")
ax.plot(metrics_hist["train_accuracy"], alpha=0.8, label="train_accuracy")
ax.plot(metrics_hist["test_accuracy"], alpha=0.8, label="test_accuracy")
plt.yscale("log")
plt.legend()
plt.savefig("trained_NNs/loss.png", dpi=300)
