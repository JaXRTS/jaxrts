"""
This submodule is dedicated to math functions which occur frequently in the calculations.
"""

from jax import numpy as jnp
import jax
from quadax import quadgk

import numpy as np


@jax.jit
def fermi_integral(x, n):
    """
    The Fermi integral of order n.
    """
    norm = jax.scipy.special.gamma(n + 1)

    def integrand(t):
        return jnp.power(t, n) / (jnp.exp(t - x[:, jnp.newaxis]) + 1)

    result, _ = quadgk(integrand, [0, jnp.inf], epsabs=1e-10, epsrel=1e-10)

    return result / norm


def numerical_inverse_fermi_integral(x, n):
    """
    WIP
    """
    pass


@jax.jit
def _R1_mk(a, b, x):

    upper_max_power = jnp.size(a)
    lower_max_power = jnp.size(b)

    powers1 = jnp.arange(0, upper_max_power, 1)[:, jnp.newaxis]
    powers2 = jnp.arange(0, lower_max_power, 1)[:, jnp.newaxis]

    numerator = jnp.sum(
        jnp.multiply(a[:, jnp.newaxis], jnp.power(x, powers1)), axis=0
    )
    denumerator = jnp.sum(
        jnp.multiply(b[:, jnp.newaxis], jnp.power(x, powers2)), axis=0
    )

    return numerator / denumerator


@jax.jit
def _F_n(x, a, b, c, d, n):
    norm = jax.scipy.special.gamma(n + 1)
    return (
        jnp.where(
            x < 2,
            jnp.exp(x) * _R1_mk(a, b, jnp.exp(x)),
            x ** (n + 1) * _R1_mk(c, d, x ** (-2)),
        )
        / norm
    )


@jax.jit
def _X_n(x, a, b, c, d, n):
    norm = jax.scipy.special.gamma(n + 1)
    x *= norm
    return jnp.where(
        x < 4 * norm,
        jnp.log(norm * x * _R1_mk(a, b, x)),
        x ** (1 / (1 + n)) * _R1_mk(c, d, x ** (-1 / (1 + n))),
    )


@jax.jit
def inverse_fermi_12_fukushima_single_prec(x):
    """
    Calculates an approximation of the Fermi integral of order 1/2 using a
    rational approximation as described in
    :cite:`Fukushima.2015`. The approximation is improved
    compared to :cite:`Antia.1993`.
    """
    norm = jax.scipy.special.gamma(1 / 2 + 1.0)
    x *= norm

    # Boundaries for the approximation
    u_sgl = [1.22, 3.43, 10.5, 34.3, 117]

    alpha = [0.0, -0.550848552, -0.482411581, -0.442839571, -0.416448071, 1.0]
    beta = [
        0.0,
        0.452114610,
        0.140636120,
        0.0420121106,
        0.0121259929,
        -111.632691,
    ]

    # u_dbl = [0.0, 1.18, 3.83, 13.4, 53.2, 188]

    r0_coeff_upper = jnp.array([127.456123, 30.3620672, 2.29733586])
    r0_coeff_lower = jnp.array([112.955041, -18.1545791, 1.0])

    r1_coeff_upper = jnp.array(
        [22.3158685, 122.487649, 135.023156, 30.5460708]
    )
    r1_coeff_lower = jnp.array([28.0566860, 59.9641578, 27.8629074, 1.0])

    r2_coeff_upper = jnp.array(
        [74.0135089, 294.367987, 294.232354, 64.9306737]
    )
    r2_coeff_lower = jnp.array([27.8728584, 62.0649704, 27.1148810, 1.0])

    r3_coeff_upper = jnp.array(
        [156.549383, 612.130579, 639.532875, 145.238686]
    )
    r3_coeff_lower = jnp.array([25.4019873, 59.3035998, 26.9857305, 1.0])

    r4_coeff_upper = jnp.array(
        [376.286772, 1449.93277, 1487.47498, 333.722383]
    )
    r4_coeff_lower = jnp.array([27.2967945, 61.1014417, 27.1844466, 1.0])

    rS_coeff_upper = jnp.array([1974.50048, 144.437558, 1.0])
    rS_coeff_lower = jnp.array([10.3906494, 0.669052603])

    term1 = jnp.where(
        x < u_sgl[0], jnp.log(x * _R1_mk(r0_coeff_upper, r0_coeff_lower, x)), 0
    )
    term2 = jnp.where(
        (x >= u_sgl[0]) * (x < u_sgl[1]),
        _R1_mk(r1_coeff_upper, r1_coeff_lower, alpha[1] + beta[1] * x),
        0,
    )
    term3 = jnp.where(
        (x >= u_sgl[1]) * (x < u_sgl[2]),
        _R1_mk(r2_coeff_upper, r2_coeff_lower, alpha[2] + beta[2] * x),
        0,
    )
    term4 = jnp.where(
        (x >= u_sgl[2]) * (x < u_sgl[3]),
        _R1_mk(r3_coeff_upper, r3_coeff_lower, alpha[3] + beta[3] * x),
        0,
    )
    term5 = jnp.where(
        (x >= u_sgl[3]) * (x < u_sgl[4]),
        _R1_mk(r4_coeff_upper, r4_coeff_lower, alpha[4] + beta[4] * x),
        0,
    )
    term6 = jnp.where(
        (x >= u_sgl[4]),
        jnp.sqrt(
            _R1_mk(
                rS_coeff_upper, rS_coeff_lower, 1.0 + beta[5] * x ** (-4 / 3)
            )
            / (-beta[5] * x ** (-4 / 3))
        ),
        0,
    )

    return term1 + term2 + term3 + term4 + term5 + term6


@jax.jit
def fermi_12_rational_approximation_antia(x):

    a = jnp.array(
        [
            5.75834152995465e6,
            1.30964880355883e7,
            1.07608632249013e7,
            3.93536421893014e6,
            6.42493233715640e5,
            4.16031909245777e4,
            7.77238678539648e2,
            1.00000000000000e0,
        ]
    )

    b = jnp.array(
        [
            6.49759261942269e6,
            1.70750501625775e7,
            1.69288134856160e7,
            7.95192647756086e6,
            1.83167424554505e6,
            1.95155948326832e5,
            8.17922106644547e3,
            9.02129136642157e1,
        ]
    )

    c = jnp.array(
        [
            4.85378381173415e-14,
            1.64429113030738e-11,
            3.76794942277806e-09,
            4.69233883900644e-07,
            3.40679845803144e-05,
            1.32212995937796e-03,
            2.60768398973913e-02,
            2.48653216266227e-01,
            1.08037861921488e00,
            1.91247528779676e00,
            1.00000000000000e00,
        ]
    )

    d = jnp.array(
        [
            7.28067571760518e-14,
            2.45745452167585e-11,
            5.62152894375277e-09,
            6.96888634549649e-07,
            5.02360015186394e-05,
            1.92040136756592e-03,
            3.66887808002874e-02,
            3.24095226486468e-01,
            1.16434871200131e00,
            1.34981244060549e00,
            2.01311836975930e-01,
            -2.14562434782759e-02,
        ]
    )

    return _F_n(x, a, b, c, d, 0.5)


@jax.jit
def fermi_neg12_rational_approximation_antia(x):

    a = jnp.array(
        [
            1.71446374704454e7,
            3.88148302324068e7,
            3.16743385304962e7,
            1.14587609192151e7,
            1.83696370756153e6,
            1.14980998186874e5,
            1.98276889924768e3,
            1.00000000000000e0,
        ]
    )

    b = jnp.array(
        [
            9.67282587452899e6,
            2.87386436731785e7,
            3.26070130734158e7,
            1.77657027846367e7,
            4.81648022267831e6,
            6.13709569333207e5,
            3.13595854332114e4,
            4.35061725080755e2,
        ]
    )

    c = jnp.array(
        [
            -4.46620341924942e-15,
            -1.58654991146236e-12,
            -4.44467627042232e-10,
            -6.84738791621745e-8,
            -6.64932238528105e-6,
            -3.69976170193942e-4,
            -1.12295393687006e-2,
            -1.60926102124442e-1,
            -8.52408612877447e-1,
            -7.45519953763928e-1,
            2.98435207466372e00,
            1.0,
        ]
    )

    d = jnp.array(
        [
            -2.23310170962369e-15,
            -7.94193282071464e-13,
            -2.22564376956228e-10,
            -3.43299431079845e-08,
            -3.33919612678907e-06,
            -1.86432212187088e-04,
            -5.69764436880529e-03,
            -8.34904593067194e-02,
            -4.78770844009440e-01,
            -4.99759250374148e-01,
            1.86795964993052e00,
            4.16485970495288e-01,
        ]
    )

    return _F_n(x, a, b, c, d, -0.5)


@jax.jit
def fermi_32_rational_approximation_antia(x):
    a = jnp.array(
        [
            4.32326386604283e4,
            8.55472308218786e4,
            5.95275291210962e4,
            1.77294861572005e4,
            2.21876607796460e3,
            9.90562948053193e1,
            1.00000000000000e0,
        ]
    )

    b = jnp.array(
        [
            3.25218725353467e4,
            7.01022511904373e4,
            5.50859144223638e4,
            1.95942074576400e4,
            3.20803912586318e3,
            2.20853967067789e2,
            5.05580641737527e0,
            1.99507945223266e02,
        ]
    )

    c = jnp.array(
        [
            2.80452693148553e-13,
            8.60096863656367e-11,
            1.62974620742993e-08,
            1.63598843752050e-06,
            9.12915407846722e-05,
            2.62988766922117e-03,
            3.85682997219346e-02,
            2.78383256609605e-01,
            9.02250179334496e-01,
            1.00000000000000e00,
        ]
    )

    d = jnp.array(
        [
            7.01131732871184e-13,
            2.10699282897576e-10,
            3.94452010378723e-08,
            3.84703231868724e-06,
            2.04569943213216e-04,
            5.31999109566385e-03,
            6.39899717779153e-02,
            3.14236143831882e-01,
            4.70252591891375e-01,
            -2.15540156936373e-02,
            2.34829436438087e-03,
        ]
    )
    return _F_n(x, a, b, c, d, 1.5)


@jax.jit
def fermi_52_rational_approximation_antia(x):
    a = jnp.array(
        [
            6.61606300631656e4,
            1.20132462801652e5,
            7.67255995316812e4,
            2.10427138842443e4,
            2.44325236813275e3,
            1.02589947781696e2,
            1.00000000000000e0,
        ]
    )

    b = jnp.array(
        [
            1.99078071053871e4,
            3.79076097261066e4,
            2.60117136841197e4,
            7.97584657659364e3,
            1.10886130159658e3,
            6.35483623268093e1,
            1.16951072617142e0,
            3.31482978240026e-3,
        ]
    )

    c = jnp.array(
        [
            8.42667076131315e-12,
            2.31618876821567e-09,
            3.54323824923987e-07,
            2.77981736000034e-05,
            1.14008027400645e-03,
            2.32779790773633e-02,
            2.39564845938301e-01,
            1.24415366126179e00,
            3.18831203950106e00,
            3.42040216997894e00,
            1.00000000000000e00,
        ]
    )

    d = jnp.array(
        [
            2.94933476646033e-11,
            7.68215783076936e-09,
            1.12919616415947e-06,
            8.09451165406274e-05,
            2.81111224925648e-03,
            3.99937801931919e-02,
            2.27132567866839e-01,
            5.31886045222680e-01,
            3.70866321410385e-01,
            2.27326643192516e-02,
        ]
    )
    return _F_n(x, a, b, c, d, 2.5)


@jax.jit
def inverse_fermi_12_rational_approximation_antia(x):
    """
    Calculates F_{1/2} using a rational function approximation as described in Antia 1993.

    """

    # Set parameters of the approximating functions (see Antia 1993 eq. (6)) for fixed k_1 = m_1 = 7.

    a = jnp.array(
        [
            1.999266880833e4,
            5.702479099336e3,
            6.610132843877e2,
            3.818838129486e1,
            1.000000000000,
        ]
    )

    # a_2 = jnp.array([4.4593646E+01,
    #             1.1288764E+01,
    #             1.0000000E+00])

    # b_2 = jnp.array([3.9519346E+01,
    #                 -5.7517464E+00,
    #                 2.6594291E-01])

    b = jnp.array(
        [
            1.771804140488e4,
            -2.014785161019e3,
            9.130355392717e1,
            -1.670718177489e0,
        ]
    )

    c = jnp.array(
        [
            -1.277060388085e-2,
            7.187946804945e-2,
            -4.262314235106e-1,
            4.997559426872e-1,
            -1.285579118012e0,
            -3.930805454272e-1,
            1.000000000000e0,
        ]
    )

    c = jnp.array([3.4873722e01, -2.6922515e01, 1.0000000e00])

    d = jnp.array([2.6612832e01, -2.0452930e01, 1.1808945e01])
    # d = jnp.array([-9.745794806288E-3,
    #                 5.485432756838E-2,
    #                 -3.299466243260E-1,
    #                 4.077841975923E-1,
    #                 -1.145531476975E0,
    #                 -6.067091689181E-2])

    return _X_n(x, a, b, c, d, 0.5)
