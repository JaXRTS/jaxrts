Saving and loading objects
==========================

*jaxrts* allows saving and restoring :py:class:`jaxrts.plasma_state.PlasmaState`,
:py:class:`jaxrts.setup.Setup` and others to / from the disk.

The underlying functionality is implemented in :py:mod:`jaxrts.saving`, and relies
on converting objects into serializable dictionaries. The results are stored as
`.json` files.

Note that the preferred way to construct a :py:class:`PlasmaState` or other
objects is by just doing so programmatically. Many examples for doing so are
given in the :doc:`example gallery<gen_examples/index>`.


.. warning::

   Saved files are **not guaranteed to be forward-compatible**, as you just
   store the instructions how to build an object, and not the object itself.


Saving an object
----------------

The following code write a :py:class:`jaxrts.plasma_state.PlasmaState` to the
file ``state.json``.

.. code:: python

   from jax import numpy as jnp
   import jaxrts
   ureg = jaxrts.ureg

   state = jaxrts.PlasmaState(
       ions=[jaxrts.Element("C")],
       Z_free=jnp.array([3.5]),
       mass_density=jnp.array([3.5]) * ureg.gram / ureg.centimeter**3,
       T_e=10 * ureg.electron_volt / ureg.k_B,
   )

   with open("state.json", "w") as f:
       jaxrts.saving.dump(state, f, indent=2)



.. collapse:: Content of state.json

   .. code:: json

      {
        "_type": "PlasmaState",
        "value": [
          {
            "Z_free": {
              "_type": "Array",
              "value": [
                3.5
              ]
            },
            "mass_density": {
              "_type": "Quantity",
              "value": [
                {
                  "_type": "Array",
                  "value": [
                    3.5
                  ]
                },
                [
                  [
                    "gram",
                    1
                  ],
                  [
                    "centimeter",
                    -3
                  ]
                ]
              ]
            },
            "T_e": {
              "_type": "Quantity",
              "value": [
                10.0,
                [
                  [
                    "electron_volt",
                    1
                  ],
                  [
                    "boltzmann_constant",
                    -1
                  ]
                ]
              ]
            },
            "T_i": {
              "_type": "Quantity",
              "value": [
                {
                  "_type": "Array",
                  "value": [
                    10.0
                  ]
                },
                [
                  [
                    "electron_volt",
                    1
                  ],
                  [
                    "boltzmann_constant",
                    -1
                  ]
                ]
              ]
            },
            "ion_core_radius": {
              "_type": "Quantity",
              "value": [
                {
                  "_type": "Array",
                  "value": [
                    -1.0
                  ]
                },
                [
                  [
                    "angstrom",
                    1
                  ]
                ]
              ]
            },
            "models": {
              "screening length": {
                "_type": "Model",
                "value": [
                  "DebyeHueckelScreeningLength",
                  [
                    [],
                    [
                      "screening length"
                    ]
                  ]
                ]
              },
              "ee-lfc": {
                "_type": "Model",
                "value": [
                  "LFCConstant",
                  [
                    [
                      0.0
                    ],
                    [
                      "ee-lfc"
                    ]
                  ]
                ]
              },
              "free-bound scattering": {
                "_type": "Model",
                "value": [
                  "Neglect",
                  [
                    [],
                    [
                      "free-bound scattering"
                    ]
                  ]
                ]
              }
            }
          },
          {
            "ions": [
              {
                "_type": "Element",
                "value": "C"
              }
            ]
          }
        ]
      }


This file contains all the information required to create the plasma state.
Apart from Callables, which are stored using the `dill
<https://github.com/uqfoundation/dill>`__ packge
:cite:`McKerns.2010,McKerns.2011`, the file is human-readable and could be
edited.

Loading an object
-----------------

.. warning::

   Similar to ``pickle``, ``dill`` is not secure. Load only data you trust. In
   regard to ``jaxrts``, this especially means that you should only load a
   :py:class:`jaxrts.setup.Setup` from trustworthy sources, because the
   :py:attr:`jaxrts.setup.Setup.instrument` is stored using ``dill``.


To load a plasma state, we also have to specify the
`:py:class:jpu.UnitRegistry` instance, that should be used. This should be
`jaxrts.ureg`.


.. code:: Python

   with open("state.json") as f:
       state = jaxrts.saving.load(f, unit_reg = jaxrts.ureg)

    print(state)


.. code:: none

   ions                     :       C
   mass densities (g/cc)    :    3.50
   ionization               :    3.50
   ion temperatures (eV)    :    10.0
   e- temperature (eV)      :    10.0

   Models attached
   ===============
   screening length         : DebyeHueckelScreeningLength
   ee-lfc                   : LFCConstant
   free-bound scattering    : Neglect


.. warning::

   Loading requires that all referenced :py:class:`jaxrts.models.Model` classes
   are known to the importer. If custom models were used, they have to be
   registered by using the ``additional_mappings`` argument of
   :py:func:`jaxrts.saving.load`.
