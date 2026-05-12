Creating your own model
=======================

In jaxrts, we strongly encourage community-driven collaboration.
Users are welcome to create, implement, or contribute new models to the codebase, targeting specific components of the simulation.
This section provides a brief overview of how to implement a new model within the framework.

To showcase this, we’ll demonstrate how to create a simple, hypothetical model for the **Local Field correction** (LFC) that always returns :math:`\pi`, for any plasma conditions.

First, create a low-level function that calculates the respective quantity from a set of parameters

.. code-block:: Python

   import jax.numpy as jnp
   from .units import Quantity
   ...


      def lfc_pi(
         Zi: float,
         ne: Quantity,
         ni: Quantity,
         Te: Quantity,
         Ti: Quantity,
         Zbar: float | None = None
      ) -> Quantity:

      return jnp.pi

Next, create the corresponding :py:class:`jaxrts.models.Model` class. Use the existing implementations for the same model
type as a reference for structure and style.

.. code-block:: Python

   import jax
   import jaxrts
   from jaxrts.setup import Setup
   ...

   class PiLFC(jaxrts.models.Model):
      """
      Hypothetical LFC Model, in which the LFC is always pi.
      """

      # The allowed model keys for a plasma state.
      allowed_keys = ["ee-lfc"]
      __name__ = "PiLFC"

      # Citations keys for reference
      cite_keys = ["JohnDoe.2025"]

      def __init__(self):
         super().__init__()

      # This function is required by every model.
      @jax.jit
      def evaluate(self, plasma_state: "PlasmaState", setup: Setup) -> Quantity:
         return lfc_pi(
               plasma_state.Z_free,
               plasma_state.n_e,
               plasma_state.n_i,
               plasma_state.T_e,
               plasma_state.T_i,
               Zbar=plasma_state.Z_free
         )

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
       PiLFC,
       PiLFC._tree_flatten,
       PiLFC._tree_unflatten,
   )

Congratulations -- you’ve successfully created your own model! You can add it to a :py:class:`jaxrts.plasmastate.PlasmaState` by assigning it to the ``"ee-lfc"`` key:

.. code-block:: Python

   state["ee-lfc"] = PiLFC()

.. note::

   A :py:class:`jaxrts.models.Model` provides the :py:meth:`jaxrts.models.Model.check` and :py:meth:`jaxrts.models.Model.prepare` methods. The former should be used to raise errors e.g., if the model is only applicable to one component systems. The other can be used in order to modify the passed :py:class:`jaxrts.plasmastate.PlasmaState`, e.g., to set sane defaults for other, subsequent Models using :py:meth:`jaxrts.plasmastate.PlasmaState.update_default_model`.


Subclasses of Model
-------------------

As some :py:class:`jaxrts.models.Model` types require further methods, :py:mod:`jaxrts.models` provides subclasses for specific keys / models. Find a list bellow:

* :py:class:`jaxrts.models.ScatteringModel`, the class for inelastic scattering models. These models have to implement an :py:meth:`evaluate_raw() <jaxrts.models.ScatteringModel.evaluate_raw>` model over :py:meth:`evaluate() <jaxrts.models.ScatteringModel.evaluate>`. The former should implement a dynamic structure factor. Convolution with the SIF (:py:attr:`jaxrts.setup.Setup.instrument`) is then handled automatically when :py:meth:`evaluate() <jaxrts.models.ScatteringModel.evaluate>` is called.

    * :py:class:`jaxrts.models.FreeFreeModel` further subclassing :py:class:`jaxrts.models.ScatteringModel`, these models additionally have to provide a :py:meth:`susceptibility() <jaxrts.models.FreeFreeModel.susceptibility>` method to expose to other models.

* :py:class:`jaxrts.models.IonFeatModel`, the class for ion feature / elastic scattering models. These models have to implement an :py:meth:`S_ii() <jaxrts.models.IonFeatModel.S_ii>` model over :py:meth:`evaluate() <jaxrts.models.IonFeatModel.evaluate>`. The former should implement a static structure factor for the ions. Convolution with the SIF (:py:attr:`jaxrts.setup.Setup.instrument`) and calculation of the :py:meth:`Rayleigh_weight() <jaxrts.models.IonFeatModel.Rayleigh_weight>` is then handled automatically when :py:meth:`evaluate() <jaxrts.models.IonFeatModel.evaluate>` is called.
* :py:class:`jaxrts.models.IPDModel`, for models for the ionization potentials depression (key ``"idp"``). Allows to add Models to a :py:class:`jaxrts.models.IPDSum`. These models have to provide the :py:meth:`all_element_states() <jaxrts.models.IPDModel.all_element_states>` method, returning the value of the IPD for all possible ionization states of the plasma constitutes.
* :py:class:`jaxrts.models.BM_V_eiSModel`, for the electron-ion potentials evaluated when considering Born-Mermin style Models. Should implement a :py:meth:`V() <jaxrts.models.BM_V_eiSModel.V>` method over :py:meth:`evaluate() <jaxrts.models.BM_V_eiSModel.evaluate>`
