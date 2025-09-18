Models implemented
==================

        This page shows an automatically generated overview over all models
        defined in :py:mod:`jaxrts.models` and :py:mod:`jaxrts.hnc_potentials`.
        The latter module contains only the potentials relevant for calculating
        the elastic scattering in the Hypernetted Chain approach.


        The following keys are available to add to
        :py:class:`jaxrts.plasmastate.PlasmaState`:

        ``screening length``, ``ionic scattering``, ``BM S_ii``, ``Debye temperature``, ``free-free scattering``, ``chemical potential``, ``ipd``, ``screening``, ``BM V_eiS``, ``free-bound scattering``, ``ee-lfc``, ``bound-free scattering``, ``form-factors``, ``ion-ion Potential``, ``electron-ion Potential``, ``electron-electron Potential``

        To set a specific model, add it to a
        :py:class:`jaxrts.plasmastate.PlasmaState`,
        e.g.,

        >>> state["free-free scattering"] = jaxrts.models.RPA_DandreaFit()

        

screening length
----------------
.. autosummary::
    :toctree: _autosummary
    :recursive:

    jaxrts.models.ArbitraryDegeneracyScreeningLength
    jaxrts.models.ConstantScreeningLength
    jaxrts.models.DebyeHueckelScreeningLength
    jaxrts.models.Gericke2010ScreeningLength


ionic scattering
----------------
.. autosummary::
    :toctree: _autosummary
    :recursive:

    jaxrts.models.ArkhipovIonFeat
    jaxrts.models.DebyeWallerSolid
    jaxrts.models.FixedSii
    jaxrts.models.Gregori2003IonFeat
    jaxrts.models.Gregori2006IonFeat
    jaxrts.models.Neglect
    jaxrts.models.OnePotentialHNCIonFeat
    jaxrts.models.PeakCollection
    jaxrts.models.ThreePotentialHNCIonFeat


BM S_ii
-------

When calculating the the Born collision frequencies in :py:class:`jaxrts.models.BornMermin` and derived free-free scattering models, one needs a notion of the static ionic structure factor. For a single species, :py:class:`jaxrts.models.Sum_Sii` is identical to the model used for the ``ionic scattering`` key. However, it is not clear if the sum calculated there also holds true for a multi-species plasma, where we would run an HNC calculation, which cannot be directly used. In this case, :py:class:`jaxrts.models.AverageAtom_Sii` might be a reasonable alternative, as it calculates S_ii in an HNC step for an average atom.

.. autosummary::
    :toctree: _autosummary
    :recursive:

    jaxrts.models.AverageAtom_Sii
    jaxrts.models.Sum_Sii


Debye temperature
-----------------
.. autosummary::
    :toctree: _autosummary
    :recursive:

    jaxrts.models.BohmStaver
    jaxrts.models.ConstantDebyeTemp


free-free scattering
--------------------
.. autosummary::
    :toctree: _autosummary
    :recursive:

    jaxrts.models.BornMermin
    jaxrts.models.BornMerminFull
    jaxrts.models.BornMermin_Fit
    jaxrts.models.BornMermin_Fortmann
    jaxrts.models.Neglect
    jaxrts.models.QCSalpeterApproximation
    jaxrts.models.RPA_DandreaFit
    jaxrts.models.RPA_NoDamping


chemical potential
------------------
.. autosummary::
    :toctree: _autosummary
    :recursive:

    jaxrts.models.ConstantChemPotential
    jaxrts.models.IchimaruChemPotential


ipd
---
.. autosummary::
    :toctree: _autosummary
    :recursive:

    jaxrts.models.ConstantIPD
    jaxrts.models.DebyeHueckelIPD
    jaxrts.models.EckerKroellIPD
    jaxrts.models.IonSphereIPD
    jaxrts.models.Neglect
    jaxrts.models.PauliBlockingIPD
    jaxrts.models.StewartPyattIPD


screening
---------
.. autosummary::
    :toctree: _autosummary
    :recursive:

    jaxrts.models.DebyeHueckelScreening
    jaxrts.models.FiniteWavelengthScreening
    jaxrts.models.Gregori2004Screening
    jaxrts.models.LinearResponseScreening
    jaxrts.models.LinearResponseScreeningGericke2010


BM V_eiS
--------

These models implement potentials which can be when calculating the Born collision frequencies in :py:class:`jaxrts.models.BornMermin` and derived free-free scattering models.

.. autosummary::
    :toctree: _autosummary
    :recursive:

    jaxrts.models.DebyeHueckel_BM_V
    jaxrts.models.FiniteWavelength_BM_V


free-bound scattering
---------------------
.. autosummary::
    :toctree: _autosummary
    :recursive:

    jaxrts.models.DetailedBalance
    jaxrts.models.Neglect


ee-lfc
------
.. autosummary::
    :toctree: _autosummary
    :recursive:

    jaxrts.models.ElectronicLFCConstant
    jaxrts.models.ElectronicLFCGeldartVosko
    jaxrts.models.ElectronicLFCStaticInterpolation
    jaxrts.models.ElectronicLFCUtsumiIchimaru


bound-free scattering
---------------------
.. autosummary::
    :toctree: _autosummary
    :recursive:

    jaxrts.models.Neglect
    jaxrts.models.SchumacherImpulse
    jaxrts.models.SchumacherImpulseFitRk


form-factors
------------
.. autosummary::
    :toctree: _autosummary
    :recursive:

    jaxrts.models.PaulingFormFactors


ion-ion Potential
-----------------
.. autosummary::
    :toctree: _autosummary
    :recursive:

    jaxrts.hnc_potentials.CoulombPotential
    jaxrts.hnc_potentials.DebyeHueckelPotential
    jaxrts.hnc_potentials.DeutschPotential
    jaxrts.hnc_potentials.HNCPotential
    jaxrts.hnc_potentials.KelbgPotential
    jaxrts.hnc_potentials.PauliClassicalMap
    jaxrts.hnc_potentials.PotentialSum
    jaxrts.hnc_potentials.ScaledPotential
    jaxrts.hnc_potentials.SpinAveragedEEExchange
    jaxrts.hnc_potentials.SpinSeparatedEEExchange


electron-ion Potential
----------------------
.. autosummary::
    :toctree: _autosummary
    :recursive:

    jaxrts.hnc_potentials.CoulombPotential
    jaxrts.hnc_potentials.DebyeHueckelPotential
    jaxrts.hnc_potentials.DeutschPotential
    jaxrts.hnc_potentials.EmptyCorePotential
    jaxrts.hnc_potentials.HNCPotential
    jaxrts.hnc_potentials.KelbgPotential
    jaxrts.hnc_potentials.KlimontovichKraeftPotential
    jaxrts.hnc_potentials.PauliClassicalMap
    jaxrts.hnc_potentials.PotentialSum
    jaxrts.hnc_potentials.ScaledPotential
    jaxrts.hnc_potentials.SoftCorePotential
    jaxrts.hnc_potentials.SpinAveragedEEExchange
    jaxrts.hnc_potentials.SpinSeparatedEEExchange


electron-electron Potential
---------------------------
.. autosummary::
    :toctree: _autosummary
    :recursive:

    jaxrts.hnc_potentials.CoulombPotential
    jaxrts.hnc_potentials.DebyeHueckelPotential
    jaxrts.hnc_potentials.DeutschPotential
    jaxrts.hnc_potentials.HNCPotential
    jaxrts.hnc_potentials.KelbgPotential
    jaxrts.hnc_potentials.PauliClassicalMap
    jaxrts.hnc_potentials.PotentialSum
    jaxrts.hnc_potentials.ScaledPotential
    jaxrts.hnc_potentials.SpinAveragedEEExchange
    jaxrts.hnc_potentials.SpinSeparatedEEExchange
