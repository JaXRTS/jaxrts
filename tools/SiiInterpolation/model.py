"""
Define the NN layout, and also create a jaxrts.Model for easily use a trained
NN with jaxrts.
"""

import json

from flax import nnx
import jaxrts
import jax
import jax.numpy as jnp
import jpu.numpy as jnpu
import orbax.checkpoint as ocp

ureg = jaxrts.ureg


class NNModel(nnx.Module):
    def __init__(self, din, dhid, dout, rngs: nnx.Rngs, no_of_atoms=1):
        self.linears = [nnx.Linear(din, dhid[0], rngs=rngs)]
        for i in range(len(dhid) - 1):
            self.linears.append(nnx.Linear(dhid[i], dhid[i + 1], rngs=rngs))
        self.linears.append(nnx.Linear(dhid[-1], dout, rngs=rngs))

        # These are the norming values for the input layer of the model.
        # We store them here to be able to obtain physical quantities from
        # scaled ones. For starters, set all values to unity. When the training
        # data is defined, replace these values.
        self.norm_theta = nnx.Variable(1.0)
        self.norm_rho = nnx.Variable(1.0)
        self.norm_Z = nnx.Variable([1.0] * no_of_atoms)
        self.norm_k_over_qk = nnx.Variable(1.0)

    @nnx.jit
    def __call__(self, x):
        for lin in self.linears[:-1]:
            x = lin(x)
            x = nnx.relu(x)
        x = self.linears[-1](x)
        return x

    def set_norms(self, theta, rho, Z, k_over_qk):
        self.norm_theta = nnx.Variable(theta)
        self.norm_rho = nnx.Variable(rho)
        self.norm_Z = nnx.Variable(Z)
        self.norm_k_over_qk = nnx.Variable(k_over_qk)

    @property
    def norms(self):
        return {
            "theta": self.norm_theta,
            "rho": self.norm_rho,
            "Z": self.norm_Z,
            "k_over_qk": self.norm_k_over_qk,
        }

    @property
    def din(self):
        """
        Number of input nodes
        """
        return self.linears[0].in_features

    @property
    def dhid(self):
        """
        Size of the hidden nodes
        """
        return [layer.in_features for layer in self.linears[1:]]

    @property
    def dout(self):
        """
        Size of the hidden nodes
        """
        return self.linears[-1].out_features

    @property
    def shape(self):
        return self.din, self.dhid, self.dout


sharding = jax.sharding.NamedSharding(
    jax.sharding.Mesh(jax.devices(), ("x",)),
    jax.sharding.PartitionSpec(),
)


def set_sharding(x: jax.ShapeDtypeStruct) -> jax.ShapeDtypeStruct:
    x.sharding = sharding
    return x


class NNSiiModel(jaxrts.models.IonFeatModel):
    def __init__(self, checkpoint_dir):
        # This requires that the network shape was saved in the checkpoint
        # directory. This is done in a slightly hacky way, see the train.py
        # file.
        with open(checkpoint_dir / "SHAPE") as f:
            shape = json.load(f)
        shape.update({"rngs": nnx.Rngs(0)})
        abstract_model = nnx.eval_shape(lambda: NNModel(**shape))

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

    # The following is required to jit a Model
    def _tree_flatten(self):
        children = (self.graphdef, self.model_state)
        aux_data = (self.model_key,)  # static values
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        obj = object.__new__(cls)
        (obj.model_key,) = aux_data
        (obj.graphdef, obj.model_state) = children
        return obj


class OneComponentNNModel(NNSiiModel):
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

        x = jnp.array(
            [
                theta / nn_model.norm_theta,
                jnp.sum(
                    plasma_state.mass_density.m_as(
                        ureg.gram / ureg.centimeter**3
                    )
                )
                / nn_model.norm_rho,
                jnp.sum(plasma_state.number_fraction * plasma_state.Z_free)
                / nn_model.norm_Z[0],
                (setup.k * (1 * ureg.hbar) / q_k).m_as(ureg.dimensionless)
                / nn_model.norm_k_over_qk,
            ]
        )

        Sii = nn_model(x)

        return jnp.array([Sii]) * ureg.dimensionless


class TwoComponentNNModel(NNSiiModel):
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

        # Catch an ionization_exanded plasma state:
        # Re-calculate plasma_state mean ionization
        unique_elements = list(set(plasma_state.ions))
        indices_element_0 = [
            i
            for i, x in enumerate(plasma_state.ions)
            if x == unique_elements[0]
        ]
        indices_element_1 = [
            i
            for i, x in enumerate(plasma_state.ions)
            if x == unique_elements[1]
        ]
        Z0 = 0

        n0 = 0
        for idx in indices_element_0:
            Z0 += plasma_state.number_fraction[idx] * plasma_state.Z_free[idx]
            n0 += plasma_state.number_fraction[idx]
        Z0 /= n0

        Z1 = 0
        n1 = 0
        for idx in indices_element_1:
            Z1 += plasma_state.number_fraction[idx] * plasma_state.Z_free[idx]
            n1 += plasma_state.number_fraction[idx]
        Z1 /= n1

        # Norm the inputs to the NN

        x = jnp.array(
            [
                theta / nn_model.norm_theta,
                jnp.sum(
                    plasma_state.mass_density.m_as(
                        ureg.gram / ureg.centimeter**3
                    )
                )
                / nn_model.norm_rho,
                Z0 / nn_model.norm_Z[0],
                Z1 / nn_model.norm_Z[1],
                (setup.k * (1 * ureg.hbar) / q_k).m_as(ureg.dimensionless)
                / nn_model.norm_k_over_qk,
            ]
        )

        Sii = nn_model(x)

        return (
            jnp.array([[Sii[0], Sii[1]], [Sii[1], Sii[2]]])
            * ureg.dimensionless
        )


_models = [OneComponentNNModel, TwoComponentNNModel]

for _m in _models:
    jax.tree_util.register_pytree_node(
        _m,
        _m._tree_flatten,
        _m._tree_unflatten,
    )
