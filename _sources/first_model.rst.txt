Creating your own model
=======================

In jaxrts, we strongly encourage community-driven collaboration.
Users are welcome to create, implement, or contribute new models to the codebase, targeting specific components of the simulation. 
This section provides a brief overview of how to implement a new model within the framework.

To showcase this, we’ll demonstrate how to create a simple, hypothetical model for **Ionization Potential Depression** (IPD) that always returns :math:`\pi`, measured in electron volts, for any plasma conditions.

First, create a low-level function that calculates the respective quantity from a set of parameters

.. code-block:: Python

   import jax.numpy as jnp
   from .units import Quantity, ureg
   ...


      def ipd_pi(
         Zi: float,
         ne: Quantity,
         ni: Quantity,
         Te: Quantity,
         Ti: Quantity,
         Zbar: float | None = None
      ) -> Quantity:

      return jnp.pi * ureg.electron_volt

Next, create the corresponding :py:class:`jaxrts.models.Model` class. Use the existing implementations for the same model
type as a reference for structure and style.

.. code-block:: Python

   import jax
   import jaxrts
   from jaxrts.setup import Setup
   ...

   class PiIPD(jaxrts.models.Model):
      """
      Hypothetical IPD Model, in which the IPD is always Pi, measured in electron volts.

      See Also
      --------
      jaxrts.ipd.ipd_pi
         Function used to calculate the IPD
      """

      # The allowed model keys for a plasma state.
      allowed_keys = ["ipd"]
      __name__ = "PiIPD"

      # Citations keys for reference
      cite_keys = ["JohnDoe.2025"]

      def __init__(self):
         super().__init__()

      # This function is required by every model.
      @jax.jit
      def evaluate(self, plasma_state: "PlasmaState", setup: Setup) -> Quantity:
         return ipd.ipd_pi(
               plasma_state.Z_free,
               plasma_state.n_e,
               plasma_state.n_i,
               plasma_state.T_e,
               plasma_state.T_i,
               Zbar=plasma_state.Z_free
         )

      # This is important for IPD models only.
      @jax.jit
      def all_element_states(
         self, plasma_state: "PlasmaState", ion_population=None
      ) -> list[jnp.ndarray]:
         out = []
         for idx, element in enumerate(plasma_state.ions):
               out.append(
                  jnp.array(
                     [
                           ipd.ipd_debye_hueckel(
                              Z,
                              plasma_state.n_e,
                              plasma_state.n_i,
                              plasma_state.T_e,
                              plasma_state.T_i,
                              Zbar=plasma_state.Z_free,
                           )[idx].m_as(ureg.electron_volt)
                           for Z in jnp.arange(element.Z)
                     ]
                  )
                  * ureg.electron_volt
               )
         return out

      # The following is required to jit a Model
      # Here, 'children' are attributes of the class that can be traced
      # using jax, e.g. plasma_state's, floats etc., while aux_data are static arguments.
      def _tree_flatten(self):
         children = ()
         aux_data = (self.model_key)  # static values
         return (children, aux_data)

      @classmethod
      def _tree_unflatten(cls, aux_data, children):
         obj = object.__new__(cls)
         (obj.model_key) = aux_data
         return obj

Finally, register your new model by calling

.. code-block:: Python

      jax.tree_util.register_pytree_node(
        PiModel,
        PiModel._tree_flatten,
        PiModel._tree_unflatten,
      )

Congratulations—you’ve successfully created your own model!

.. note::

   A :py:class:`jaxrts.models.Model` provides the :py:meth:`jaxrts.models.Model.check` and :py:meth:`jaxrts.models.Model.prepare` methods. The former should be used to raise errors e.g., if the model is only applicable to one component systems. The other can be used in order to modify the passed :py:class:`jaxrts.plasmastate.PlasmaState`, e.g., to set sane defaults for other, subsequent Models using :py:meth:`jaxrts.plasmastate.PlasmaState.update_default_model`.