"""
Based on
https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial2/Introduction_to_JAX.html
and the FLAX examples.
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

from model import NNModel

model = NNModel(5, [64, 1024, 1024], 3, nnx.Rngs(int(1e6 * time.time()) % 42))

optimizer = nnx.Optimizer(model, optax.adam(1e-3))


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


class Dataset1C(data.Dataset):
    def __init__(self, hdf5_file_path, spec1):
        with h5py.File(hdf5_file_path, "r") as hdf5_file:
            self.theta = hdf5_file["inputs"]["theta"][:]
            self.rho = hdf5_file["inputs"]["rho"][:]
            self.Z1 = hdf5_file["inputs"][f"Z_{spec1}"][:]
            self.k_over_qk = hdf5_file["inputs"]["k_over_qk"][:]
            self.S_11 = hdf5_file["outputs"][f"S_{spec1}{spec1}"][:]

        mask = ~jnp.isnan(self.S_11)

        self.theta = self.theta[mask]
        self.rho = self.rho[mask]
        self.Z1 = self.Z1[mask]
        self.k_over_qk = self.k_over_qk[mask]
        self.S_11 = self.S_11[mask]

        self.sf_theta = jnp.max(self.theta)
        self.sf_rho = jnp.max(self.rho)
        self.sf_Z1 = jnp.max(self.Z1)
        self.sf_k_over_qk = jnp.max(self.k_over_qk)

        self.s_theta = self.theta / self.sf_theta
        self.s_rho = self.rho / self.sf_rho
        self.s_Z1 = self.Z1 / self.sf_Z1
        self.s_k_over_qk = self.k_over_qk / self.sf_k_over_qk

    def __len__(self):
        return len(self.S_11)

    def __getitem__(self, idx):
        inputs = [
            self.s_theta[idx],
            self.s_rho[idx],
            self.s_Z1[idx],
            self.s_k_over_qk[idx],
        ]
        outputs = [
            self.S_11[idx],
        ]

        return onp.array(inputs), onp.array(outputs)


class Dataset2C(data.Dataset):
    def __init__(self, hdf5_file_path, spec1, spec2):
        with h5py.File(hdf5_file_path, "r") as hdf5_file:
            self.theta = hdf5_file["inputs"]["theta"][:]
            self.rho = hdf5_file["inputs"]["rho"][:]
            self.Z1 = hdf5_file["inputs"][f"Z_{spec1}"][:]
            self.Z2 = hdf5_file["inputs"][f"Z_{spec2}"][:]
            self.k_over_qk = hdf5_file["inputs"]["k_over_qk"][:]
            self.S_11 = hdf5_file["outputs"][f"S_{spec1}{spec1}"][:]
            self.S_12 = hdf5_file["outputs"][f"S_{spec1}{spec2}"][:]
            self.S_22 = hdf5_file["outputs"][f"S_{spec2}{spec2}"][:]

        mask = (
            (~jnp.isnan(self.S_22))
            & (~jnp.isnan(self.S_12))
            & (~jnp.isnan(self.S_11))
        )

        self.theta = self.theta[mask]
        self.rho = self.rho[mask]
        self.Z1 = self.Z1[mask]
        self.Z2 = self.Z2[mask]
        self.k_over_qk = self.k_over_qk[mask]
        self.S_11 = self.S_11[mask]
        self.S_12 = self.S_12[mask]
        self.S_22 = self.S_22[mask]

        self.sf_theta = jnp.max(self.theta)
        self.sf_rho = jnp.max(self.rho)
        self.sf_Z1 = jnp.max(self.Z1)
        self.sf_Z2 = jnp.max(self.Z2)
        self.sf_k_over_qk = jnp.max(self.k_over_qk)

        self.s_theta = self.theta / self.sf_theta
        self.s_rho = self.rho / self.sf_rho
        self.s_Z2 = self.Z2 / self.sf_Z2
        self.s_Z1 = self.Z1 / self.sf_Z1
        self.s_k_over_qk = self.k_over_qk / self.sf_k_over_qk

    def __len__(self):
        return len(self.S_11)

    def __getitem__(self, idx):
        inputs = [
            self.s_theta[idx],
            self.s_rho[idx],
            self.s_Z1[idx],
            self.s_Z2[idx],
            self.s_k_over_qk[idx],
        ]
        outputs = [
            self.S_11[idx],
            self.S_12[idx],
            self.S_22[idx],
        ]

        return onp.array(inputs), onp.array(outputs)


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


dataset = Dataset2C("water.h5", "H", "O")
# Set the norms, extracted from the dataset
model.set_norms(
    theta=float(dataset.sf_theta),
    rho=float(dataset.sf_rho),
    Z=[float(dataset.sf_Z1), float(dataset.sf_Z2)],
    k_over_qk=float(dataset.sf_k_over_qk),
)
print(model.norms)

train_dataset, test_dataset = data.random_split(dataset, [0.8, 0.2])

batch_size = 40
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


ckpt_dir = ocp.test_utils.erase_and_create_empty("/tmp/checkpoints/")
checkpointer = ocp.StandardCheckpointer()


def train_model(model, train_loader, test_loader, metrics, num_epochs=250):
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
        if (epoch_num > 0) & (epoch_num % 10 == 0):
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
            with open(checkpoint_path / "SHAPE", "w") as f:
                shape = {
                    "din": model.din,
                    "dhid": model.dhid,
                    "dout": model.dout,
                    "no_of_atoms": len(model.norm_Z),
                }
                json.dump(shape, f)

    return metrics_history


metrics_hist = train_model(model, train_loader, test_loader, metrics, 201)

fig, ax = plt.subplots(1)
ax.plot(metrics_hist["train_loss"], label="train_loss")
ax.plot(metrics_hist["train_accuracy"], label="train_accuracy")
ax.plot(metrics_hist["test_accuracy"], label="test_accuracy")
plt.yscale("log")
plt.savefig("loss.png")


t0 = time.time()
print(model(jnp.array([0.4, 0.2, 0.5, 0.65, 0.11])))
print(time.time() - t0)
t0 = time.time()
print(model(jnp.array([0.4, 0.2, 0.5, 0.65, 0.11])))
print(time.time() - t0)
