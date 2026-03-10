"""
Interpolate the Static structure factors by a neural network. In this file, we
define the NN layout, and also create a :py:class:`jaxrts.Model` which allows
for easily using a trained NN with jaxrts.

Models have to be trained, for each sample type, separately. Tools for doing so
are provided in the `tools/SiiInterpolation/` directory of the jaxrts
repository. The trained network is saved as an :py:mod:`orbax` checkpoint (with
slight additions to save properties of the net architecture).
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import jpu.numpy as jnpu
import orbax.checkpoint as ocp
from flax import nnx

import jaxrts

ureg = jaxrts.ureg


class NNModel(nnx.Module):
    """
    A class inheriting from :py:class:`nnx.Module`, adding quality of life
    features and defining normalization features for the input of the NN.
    """

    def __init__(
        self,
        din: int,
        dhid: list[int],
        dout: int,
        rngs: nnx.Rngs,
        no_of_atoms: int | None = None,
    ):
        """
        Creates the network as a set of fully connected
        :py:class:`flax.nnx.Linear` layers.

        Parameters
        ----------
        din: int
            Number of input nodes. For a typical setup, this should be ``3 +
            no_of_atoms``.
        dhid: list[int]
            List of integers, containing the size of each hidden layer.
        dout: int
            Number of output nodes. Typically, this is
            ``no_of_ions * (no_of_ions + 1)/2``.
        rngs: nnx.Rngs
            A random number generator.
        no_of_atoms: int or None, defaults to None
            Number of species in the plasma. If this value is not given,
            explicitly, it is calculated from :py:attr:`din` as ``din - 3``.
        """
        if no_of_atoms is None:
            no_of_atoms = din - 3
        self.linears = [nnx.Linear(din, dhid[0], rngs=rngs)]
        for i in range(len(dhid) - 1):
            self.linears.append(nnx.Linear(dhid[i], dhid[i + 1], rngs=rngs))
        self.linears.append(nnx.Linear(dhid[-1], dout, rngs=rngs))

        # These are the values with which we norm the quantities for the input
        # layer of the model.
        # We store them here to be able to obtain physical quantities from
        # scaled ones. For starters, set all values to unity. When the training
        # data is defined, replace these values.
        #: Normalization for theta
        self.norm_theta = nnx.Variable(1.0)
        #: Normalization for rho.
        #: This is given as a float, as quantities would be in conflict with
        #: flax. The unit is g/cm³
        self.norm_rho = nnx.Variable(1.0)
        #: Normalization the ionization (this is an array with one entry for
        #: each component.
        self.norm_Z = nnx.Variable([1.0] * no_of_atoms)
        #: Normalization for k_over_qk
        self.norm_k_over_qk = nnx.Variable(1.0)

    @nnx.jit
    def __call__(self, x):
        for lin in self.linears[:-1]:
            x = lin(x)
            x = nnx.relu(x)
        x = self.linears[-1](x)
        return x

    def set_norms(
        self, theta: float, rho: float, Z: list[float], k_over_qk: float
    ):
        """
        Set the normalization of the input layers.
        """
        self.norm_theta = nnx.Variable(theta)
        self.norm_rho = nnx.Variable(rho)
        self.norm_Z = nnx.Variable(Z)
        self.norm_k_over_qk = nnx.Variable(k_over_qk)

    @property
    def norms(self) -> dict[str, nnx.Variable]:
        """
        Get the input layer normalizations as a dictionary.
        """
        return {
            "theta": self.norm_theta,
            "rho": self.norm_rho,
            "Z": self.norm_Z,
            "k_over_qk": self.norm_k_over_qk,
        }

    @property
    def din(self) -> int:
        """
        Shape of the input layer.
        """
        return self.linears[0].in_features

    @property
    def dhid(self) -> list[int]:
        """
        Shape of the hidden layers.
        """
        return [layer.in_features for layer in self.linears[1:]]

    @property
    def dout(self) -> list[int]:
        """
        Shape of the output layer.
        """
        return self.linears[-1].out_features

    @property
    def shape(self) -> tuple[int, list[int], int]:
        """
        Shape of the full model (input, hidden, and output layers).
        """
        return self.din, self.dhid, self.dout


sharding = jax.sharding.NamedSharding(
    jax.sharding.Mesh(jax.devices(), ("x",)),
    jax.sharding.PartitionSpec(),
)


def set_sharding(x: jax.ShapeDtypeStruct) -> jax.ShapeDtypeStruct:
    y = x.update(sharding=sharding)
    return y


class NNSiiModel(jaxrts.models.IonFeatModel):
    """
    A :py:class:`jaxrts.model.IonFeatModel` to use a neural network to obtain
    ion-ion static structure factors.

    .. note::

       This is a parent class that, in itself, has no practical application, as
       it defines no :py:meth:`jaxrts.models.IonFeatModel.S_ii` method.
       This class only handles loading the neural network from an
       :py:mod:`orbax` checkpoint.

    """

    def __init__(self, checkpoint_dir: Path):
        """
        Initialize the NNSiiModel.

        Parameters
        ----------
        checkpoint_dir : Path
            The Path to the checkpoint directory.
        """  # This requires that the network shape was saved in the checkpoint
        # directory. This is done in a slightly hacky way, see the train.py
        # file.
        with open(checkpoint_dir / "SHAPE") as f:
            shape = json.load(f)
        shape.update({"rngs": nnx.Rngs(0)})
        self.model_elements = shape.pop("elements")
        model = NNModel(**shape)
        abstract_model = nnx.eval_shape(lambda: model)

        graphdef, abstract_state = nnx.split(
            abstract_model,
        )

        ckptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())

        change_sharding_abstract_state = jax.tree_util.tree_map(
            set_sharding, abstract_state
        )

        model_state = ckptr.restore(
            checkpoint_dir,
            args=ocp.args.StandardRestore(change_sharding_abstract_state),
            target=abstract_state,
        )

        self.graphdef = graphdef
        self.model_state = model_state
        super().__init__()

    @jax.jit
    def S_ii(
        self,
        plasma_state: jaxrts.PlasmaState,
        setup: jaxrts.Setup,
    ) -> jnp.ndarray:

        # This has to be done within a jax.jitted function.
        nn_model = nnx.merge(self.graphdef, self.model_state)

        E_f = jaxrts.plasma_physics.fermi_energy(plasma_state.n_e)
        q_k = jnpu.sqrt(2 * ureg.electron_mass * E_f)

        theta = (plasma_state.T_e * ureg.k_B / E_f).m_as(ureg.dimensionless)

        # Get the elements of the model and catch expanded ionization state with unique elements:
        elements = [jaxrts.Element(e.strip("+")) for e in self.model_elements]
        unique_elements = list(dict.fromkeys(elements))

        # assign idices to later match order of elements in plasma state
        idx_arr = jnp.array(
            [
                [i for i, x in enumerate(plasma_state.ions) if x == elem]
                for elem in unique_elements
            ]
        )
        n = len(unique_elements)

        # Norm the inputs to the NN
        Z_arr = jnp.zeros(n)
        n_arr = jnp.zeros(n)
        for i in range(n):
            for idx in idx_arr[i]:

                Z_arr = Z_arr.at[i].add(
                    plasma_state.number_fraction[idx]
                    * plasma_state.Z_free[idx]
                )
                n_arr = n_arr.at[i].add(plasma_state.number_fraction[idx])

            Z_arr = Z_arr.at[i].set(Z_arr[i] / n_arr[i])

        # create input array to the Neural Net
        x = jnp.array(
            [
                theta / nn_model.norm_theta,
                jnp.sum(
                    plasma_state.mass_density.m_as(
                        ureg.gram / ureg.centimeter**3
                    )
                )
                / nn_model.norm_rho,
                *(Z_arr / jnp.array(nn_model.norm_Z)),
                (setup.k * (1 * ureg.hbar) / q_k).m_as(ureg.dimensionless)
                / nn_model.norm_k_over_qk,
            ]
        )
        Sii = nn_model(x)

        # Create output array, S_base, adaptive for N components
        # for 4 components it looks like
        # S_base = jnp.array(
        #     [
        #         [Sii[0], Sii[1], Sii[2], Sii[3]],
        #         [Sii[1], Sii[4], Sii[5], Sii[6]],
        #         [Sii[2], Sii[5], Sii[7], Sii[8]],
        #         [Sii[3], Sii[6], Sii[8], Sii[9]],
        #     ]
        # )
        m = len(elements)
        iu = jnp.triu_indices(m)
        S_base = jnp.zeros((m, m), dtype=Sii.dtype)
        S_base = S_base.at[iu].set(Sii)
        S_base = S_base + jnp.triu(S_base, 1).T

        # Determine permutation order based on ion order of plasma_state
        perm = []
        for entry in idx_arr:
            perm = [*perm, *entry]
        perm = jnp.array(perm)

        # Apply the permutation to S_base to capture wronge order of plasma state elements
        # compared to order of elements used to train the NN,
        # e.g. Plasma_state element list = ["C","H"] but NN element list =["H","C"]
        # would result in wrong Sii values set for the elements
        S_out = S_base[perm][:, perm]
        return S_out * ureg.dimensionless

    # The following is required to jit a Model
    def _tree_flatten(self):
        children = (self.graphdef, self.model_state)
        aux_data = (self.model_key, self.model_elements)  # static values
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        obj = object.__new__(cls)
        (obj.model_key, obj.model_elements) = aux_data
        (obj.graphdef, obj.model_state) = children
        return obj


def _sort_func(ion):
    return ion.symbol


_models = [NNSiiModel]

for _m in _models:
    jax.tree_util.register_pytree_node(
        _m,
        _m._tree_flatten,
        _m._tree_unflatten,
    )
