from .units import ureg, Quantity

from jpu import numpy as jnpu


def plasma_frequency(electron_density: Quantity) -> Quantity:
    """
    Calculate the plasma frequency :math:`\\omega_\\text{pe}`

    .. math::
       \\omega_\\text{pe} = \\sqrt{\\frac{e^2 n_e}{\\epsilon_0 m_e}}

    where :math:`e` is the elementary charge,
    where :math:`n_e` is the electron density,
    where :math:`\\epsilon_0` is the vacuum permittivity,
    where :math:`m_e` is the electron's mass.

    Parameters
    ----------
    electron_density
        The electron density in units of 1/volume.

    Returns
    -------
    Quantity
        The plasma frequency in units of Hz.
    """
    return jnpu.sqrt(
        (ureg.elementary_charge**2 * electron_density)
        / (ureg.vacuum_permittivity * ureg.electron_mass)
    ).to(ureg.Hz)
