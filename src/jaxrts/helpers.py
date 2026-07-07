"""
Miscellaneous helper functions.
"""

import re
import logging
from functools import partialmethod, wraps
from time import time
from platformdirs import user_cache_dir
import requests
from pathlib import Path

import jax
from jax import numpy as jnp
import jpu.numpy as jnpu

from .units import Quantity, ureg, to_array

#: Typically, we return quantities that differ per orbital in an
#: :py:class:`jax.numpy.ndarray` with 10 entries, the orbitals with n<=4.
#: This dictionary contains the orbitals as keys and the corresponding indices
#: in such arrays as values.
orbital_map = {
    "1s": 0,
    "2s": 1,
    "2p": 2,
    "3s": 3,
    "3p": 4,
    "3d": 5,
    "4s": 6,
    "4p": 7,
    "4d": 8,
    "4f": 9,
    # "5s": 10,
    # "5p": 11,
    # "5d": 12,
    # "6s": 13,
    # "6d": 14,
    # "7s": 15,
}


def orbital_array(
    n1s: int | float | Quantity = 0,
    n2s: int | float | Quantity = 0,
    n2p: int | float | Quantity = 0,
    n3s: int | float | Quantity = 0,
    n3p: int | float | Quantity = 0,
    n3d: int | float | Quantity = 0,
    n4s: int | float | Quantity = 0,
    n4p: int | float | Quantity = 0,
    n4d: int | float | Quantity = 0,
    n4f: int | float | Quantity = 0,
) -> jnp.ndarray:
    """
    Create an array with entries for each orbital.

    Parameters
    ----------
    n1s, n2s, n2p, ... int | float | Quantity, default = 0
        The values for the individual orbitals.

    Returns
    -------
    jnp.ndarray
        An array containing the provided entries, sorted so that the index of a
        specific orbital can be obtained by :py:data:`~.orbital_map`.

    Examples
    --------
    >>> carbon_occupancy = orbital_array(n1s=2, n2s=2, n2p=2)
    """
    return jnp.array(
        [
            n1s,
            n2s,
            n2p,
            n3s,
            n3p,
            n3d,
            n4s,
            n4p,
            n4d,
            n4f,
        ]
    )


def invert_dict(dictionary: dict) -> dict:
    """
    Invert a dictionary, so that it's keys become values, and the values are
    the keys of the returned dict.
    """
    out_dir = {v: k for k, v in dictionary.items()}
    if len(dictionary) != len(out_dir):
        raise ValueError(
            f"Dict {dictionary} cannot be inverted because it contains non-unique entries."  # noqa: E501
        )
    return out_dir


def timer(func, custom_prefix=None, loglevel=logging.INFO):
    """
    Simple timer wrapper.
    """

    print("Starting ", func.__name__, "...\n")

    @wraps(func)
    def wrapper(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f"Executed {func.__name__!r} ...", end="")
        print(f"in {(t2 - t1):.4f} s\n")

        return result

    return wrapper


def mass_from_number_fraction(number_fractions, elements):
    """
    Calculate the mass fraction of a mixture.

    Parameters
    ----------
    number_fractions : array_like
        The number fractions of each chemical element.
    elements : list
        The masses of the respective chemical elements.

    Returns
    -------
    ndarray
        The mass fractions of the chemical elements in the mixture.

    Raises
    ------
    ValueError
        If the lengths of `number_fractions` and `elements` are not the same.

    Examples
    --------
    >>> number_fractions = [1/3, 2/3]
    >>> elements = [jaxrts.Element("C"), jaxrts.Element("H")]
    >>> calculate_mass_fraction(number_fractions, elements)
    Array([0.85627718, 0.14372282], dtype=float64)
    """
    number_fractions = jnp.asarray(number_fractions)
    masses = jnp.array([e.atomic_mass.m_as(ureg.gram) for e in elements])

    if number_fractions.shape != masses.shape:
        raise ValueError(
            "number_fractions and elements must have the same length"
        )

    # Calculate the total mass of the mixture
    total_mass = jnp.sum(number_fractions * masses)

    # Calculate the mass fraction for each element
    mass_fractions = (number_fractions * masses) / total_mass

    return mass_fractions


def mass_density_from_electron_density(
    n_e, Z, number_fractions, elements, partial=False
):
    """
    Calculate the mass density of a mixture from electron density.

    Parameters
    ----------
    n_e (electron density): scalar
        electron density of mixture
    Z (charge): array_like
        The charge of each chemical element in the plasma.
    number_fractions : array_like
        The number fractions of each chemical element.
    elements : list
        The masses of the respective chemical elements.
    partial: boolean
        default false = density of mixture, if true: density vector splitted
        for mixture is returned.

    Returns
    -------
    array_like
        The mass density of the mixture.
        When partial=True: density is splitted up in mixture-componentes by
        multiply with :py:func:`~.mass_from_number_fraction`.

    Raises
    ------
    ValueError
        If the lengths of `Z`, `number_fractions` and `elements` are not the
        same.

    Examples
    --------
    >>> n_e = 0.8e24 / ureg.cm**3
    >>> number_fractions = jnp.array([1/2, 1/2])
    >>> elements = [jaxrts.Element("C"), jaxrts.Element("H")]
    >>> Z_free = jnp.array([4.0, 1.0])
    >>> mass_density_from_electron_density(
    >>>     n_e, Z_free, number_fractions, elements
    >>> )
    <Quantity(3.4589693021231165, 'gram / centimeter ** 3')>
    >>> mass_density_from_electron_density(
    >>>     n_e, Z_free, number_fractions, elements, partial=True
    >>> )
    <Quantity([3.19115756 0.26781174], 'gram / centimeter ** 3')>
    """

    if not (len(number_fractions) == len(Z) == len(elements)):
        raise ValueError(
            "Z, number_fractions and elements must have the same length"
        )

    # model average atom in the mixture
    m = to_array([x.atomic_mass for x in elements])
    nom = jnpu.sum(m * number_fractions)
    denom = jnpu.sum(Z * number_fractions)

    rho = n_e * nom / denom

    if partial:
        mass_fraction = mass_from_number_fraction(number_fractions, elements)
        rho = mass_fraction * rho

    return rho.to(ureg.gram / ureg.cm**3)


class JittableDict(dict):
    # The following is required to jit a state
    def _tree_flatten(self):
        children = self.values()
        aux_data = self.keys()
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        obj = JittableDict.__new__(cls)
        for key, val in zip(aux_data, children, strict=False):
            obj[key] = val
        return obj


@jax.jit
def secant_extrema_finding(func, xmin, xmax, tol=1e-7, max_iter=1e5):
    """
    Use the secant method to find the extrema of a function within specified
    bounds. This is achieved by calling :py:func:`jax.grad` on the function
    func.

    Parameters
    ----------
    func : callable
        The function to minimize. It should take a single input and return a
        scalar output.
    xmin : float
        The minimum bound for the variable x.
    xmax : float
        The maximum bound for the variable x.
    tol : float, optional
        The tolerance for the stopping criteria. The default is 1e-7.
    max_iter : int, optional
        The maximum number of iterations to perform. The default is 100000.

    Returns
    -------
    float
        The x value that minimizes the function within the specified bounds.

    Examples
    --------
    >>> def example_func(x):
    ...     return (x - 2) ** 2
    >>> minimum, iter = secant_minimum_finding(example_func, 0, 4)
    >>> print(minimum)
    2.0
    """

    f = jax.grad(func)

    x0 = (xmin + xmax) / 2
    x1 = xmax

    def body_fun(state):
        x0, x1, i = state
        f0 = f(x0)
        f1 = f(x1)

        # Secant method update
        x_next = x1 - f1 * (x1 - x0) / (f1 - f0)
        x_next = jnp.clip(x_next, xmin, xmax)

        # Update the state
        return x1, x_next, i + 1

    def cond_fun(state):
        x0, x1, i = state
        f0 = f(x0)
        f1 = f(x1)
        return (jnp.abs(f0 - f1) >= tol) & (i < max_iter)

    # Initialize the state
    state = (x0, x1, 0)
    final_state = jax.lax.while_loop(cond_fun, body_fun, state)

    return final_state[1], final_state[2]


def partialclass(cls, *args, **kwds):
    """
    This is an equivalent to functools.partial, but for Classes.

    See https://stackoverflow.com/a/38911383
    """

    class NewCls(cls):
        __init__ = partialmethod(cls.__init__, *args, **kwds)

    return NewCls


@jax.jit
def bisection(
    func, a, b, tolerance=1e-4, max_iter: int = 1e4, min_iter: int = 1e2
):
    """
    Find the root of a function ``func`` between the points ``a`` and ``b`` by
    bisection.

    This is a simple implementation, without any checks guaranteeing that the
    root is found.

    Parameters
    ----------
    func
        The function of which a root should be found.
    a
        Lower end of the interval within which the root should be searched.
    b
        Upper limit of the interval within which the root should be searched.
    tolerance
        Tolerance that sets the condition to stop the iterative search. Applies
        to both the absolute value of the function, and the absolute distance
        between two consecutive candidates.
    max_iter : int
        Maximal number of steps before the loop aborts.
    min_iter : int
        Minimal number of steps the algorithm has to run before a value is
        returned.

    Returns
    -------
    One of the function ``func``.
    """

    def condition(state):
        prev_x, next_x, count = state

        return (
            (count < max_iter)
            & (jnp.abs(func(next_x)) > tolerance)
            & (jnp.abs(prev_x - next_x) > tolerance)
        ) | (count < min_iter)

    def body(state):
        a, b, i = state
        c = (a + b) / 2  # middlepoint
        bound = jnp.where(jnp.sign(func(c)) == jnp.sign(func(b)), a, b)
        return bound, c, i + 1

    initial_state = (a, b, 0)

    _, final_state, iterations = jax.lax.while_loop(
        condition, body, initial_state
    )

    return final_state, iterations


def get_cache_dir() -> Path:
    cache_dir = Path(user_cache_dir("jaxrts", "jaxrts"))
    cache_dir.mkdir(exist_ok=True)
    return cache_dir


def download_from_nist(config) -> None:
    cache_dir = get_cache_dir()
    file = cache_dir / f"{config}.csv"
    if not file.exists():
        r = requests.get(
            f"https://physics.nist.gov/cgi-bin/ASD/energy1.pl?de=0&spectrum={config}&units=1&format=2&output=0&page_size=15&multiplet_ordered=0&average_out=1&conf_out=on&term_out=on&level_out=on&g_out=on&biblio=on&temp=&submit=Retrieve+Data"  # noqa: E501
        )
        if r.status_code == 200:
            with open(file, "w") as f:
                f.write(r.text)
        else:
            raise FileNotFoundError(
                "Failed to download file from the NIST database"
            )


def read_nist_file(config) -> (jnp.ndarray, Quantity):
    """
    Read in a nist file for excited states that was downloaded by
    :py:func:`download_from_nist`. Extracts multiplicities and energy levels.
    """
    cache_dir = get_cache_dir()
    nist_file = cache_dir / f"{config}.csv"
    with open(nist_file) as f:
        lines = f.readlines()

    pattern = re.compile(
        r'''
        ,term,                               # Literal text “,term,”
        (?P<g>\d+)                           #  "g"
        ,"=""(?P<energy>\d+(?:\.\d+)?)"""    # energy in = and "
    ''',
        re.VERBOSE,
    )
    g = []
    E = []
    for line in lines:
        match = pattern.search(line)
        if match:
            g.append(int(match.group("g")))
            E.append(float(match.group("energy")))
    return jnp.array(g), jnp.array(E) * ureg.electron_volt


def cramer_solve(A, b, N_max=4):
    """
    Solve A x = b via Cramer's rule for small, fixed system sizes, avoiding the
    fixed dispatch overhead of a batched jnp.linalg.solve. Falls back to
    jnp.linalg.solve for sizes not hand-coded below.
    For small matrices, e.g., 2 component HNC, this does serve as a notable
    speedup.

    .. warning::

       Using this function if a tradeoff of speed (for small systems) over
       numerical stability. We found that for the application in HNC with few
       components, this works reasonably well. Setting ``M_max`` to 0 allows to
       bypass explicit calculation, if required.

    A: (..., N, N), b: (..., N). N must be static (known at trace time).
    Works either called directly on a batched array, or from inside a
    vmap-mapped function (where A, b appear unbatched, shape (N,N) & (N,)).

    Parameters
    ----------
    A: jnp.ndarray
        Shape (..., N, N) matrix to be inverted
    b: jnp.ndarray
        Shape (..., N) array of solutions
    N_max: int
        Maximal size of the matrix for which the hand-written inversion should
        be performed. If the size is bigger than `N_max`, fall back to
        ``jax.linalg.solve``.

    Returns
    -------
    x: jnp.narray
        Shape (..., N), so that :math:`A x = b`
    """
    N = A.shape[-1]
    # Directly return the linalg solve based
    if N > N_max:
        return jnp.linalg.solve(A, b)

    if N == 1:
        return b / A[..., 0, 0][..., None]

    if N == 2:
        a00, a01 = A[..., 0, 0], A[..., 0, 1]
        a10, a11 = A[..., 1, 0], A[..., 1, 1]
        b0, b1 = b[..., 0], b[..., 1]

        det = a00 * a11 - a01 * a10
        x0 = (a11 * b0 - a01 * b1) / det
        x1 = (-a10 * b0 + a00 * b1) / det
        return jnp.stack([x0, x1], axis=-1)

    if N == 3:
        a00, a01, a02 = A[..., 0, 0], A[..., 0, 1], A[..., 0, 2]
        a10, a11, a12 = A[..., 1, 0], A[..., 1, 1], A[..., 1, 2]
        a20, a21, a22 = A[..., 2, 0], A[..., 2, 1], A[..., 2, 2]
        b0, b1, b2 = b[..., 0], b[..., 1], b[..., 2]

        c00 = a11 * a22 - a12 * a21
        c01 = -(a10 * a22 - a12 * a20)
        c02 = a10 * a21 - a11 * a20
        c10 = -(a01 * a22 - a02 * a21)
        c11 = a00 * a22 - a02 * a20
        c12 = -(a00 * a21 - a01 * a20)
        c20 = a01 * a12 - a02 * a11
        c21 = -(a00 * a12 - a02 * a10)
        c22 = a00 * a11 - a01 * a10

        det = a00 * c00 + a01 * c01 + a02 * c02

        x0 = (c00 * b0 + c10 * b1 + c20 * b2) / det
        x1 = (c01 * b0 + c11 * b1 + c21 * b2) / det
        x2 = (c02 * b0 + c12 * b1 + c22 * b2) / det
        return jnp.stack([x0, x1, x2], axis=-1)

    if N == 4:
        a00, a01, a02, a03 = (
            A[..., 0, 0],
            A[..., 0, 1],
            A[..., 0, 2],
            A[..., 0, 3],
        )
        a10, a11, a12, a13 = (
            A[..., 1, 0],
            A[..., 1, 1],
            A[..., 1, 2],
            A[..., 1, 3],
        )
        a20, a21, a22, a23 = (
            A[..., 2, 0],
            A[..., 2, 1],
            A[..., 2, 2],
            A[..., 2, 3],
        )
        a30, a31, a32, a33 = (
            A[..., 3, 0],
            A[..., 3, 1],
            A[..., 3, 2],
            A[..., 3, 3],
        )
        b0, b1, b2, b3 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]

        t0 = a01 * a10
        t1 = a02 * a20
        t2 = a03 * a30
        t3 = a23 * a32
        t4 = a22 + a33
        t5 = a22 * a33
        t6 = a12 * a21
        t7 = a13 * a31
        t8 = t6 + t7
        t9 = a12 * a20
        t10 = a13 * a30
        t11 = a10 * a11 + t10 + t9
        t12 = a10 * a21
        t13 = a23 * a30
        t14 = a20 * a22 + t12 + t13
        t15 = a10 * a31
        t16 = a20 * a32
        t17 = a30 * a33 + t15 + t16
        t18 = a23 * a31
        t19 = a21 * a32
        t20 = a01 * a12
        t21 = a03 * a11
        t22 = a01 * a13
        t23 = a02 * a33
        t24 = a03 * a31
        t25 = a13 * a21
        t26 = a02 * a23
        t27 = a00 * a12
        t28 = a03 * a10
        t29 = a00 * a13
        t30 = a11 * a33
        t31 = a00 * a11
        t32 = a22 * a30
        t33 = a02 * a30

        det = (
            a00
            * (
                a11 * (-t3 + t5)
                + a12 * (a21 * a22 + t18)
                + a13 * (a31 * a33 + t19)
                - t4 * t8
            )
            - a01 * (a11 * t11 + a12 * t14 + a13 * t17)
            - a02 * (a21 * t11 + a22 * t14 + a23 * t17)
            - a03 * (a31 * t11 + a32 * t14 + a33 * t17)
            + (a11 + t4) * (a01 * t11 + a02 * t14 + a03 * t17)
            + (t0 + t1 + t2) * (-a11 * t4 + t3 - t5 + t8)
        )
        num0 = (
            b0
            * (
                -a11 * t3
                + a11 * t5
                + a12 * t18
                + a13 * t19
                - a22 * t7
                - a33 * t6
            )
            - b1
            * (
                -a01 * t3
                + a01 * t5
                + a02 * t18
                + a03 * t19
                - a21 * t23
                - a22 * t24
            )
            + b2
            * (
                a02 * t7
                - a11 * t23
                - a12 * t24
                + a32 * t21
                - a32 * t22
                + a33 * t20
            )
            - b3
            * (
                a02 * t25
                - a03 * t6
                - a11 * t26
                + a22 * t21
                - a22 * t22
                + a23 * t20
            )
        )
        num1 = (
            -b0
            * (
                -a10 * t3
                + a10 * t5
                + a12 * t13
                + a13 * t16
                - a22 * t10
                - a33 * t9
            )
            + b1
            * (
                -a00 * t3
                + a00 * t5
                + a02 * t13
                + a03 * t16
                - a22 * t2
                - a33 * t1
            )
            - b2
            * (
                a02 * t10
                - a10 * t23
                - a12 * t2
                + a32 * t28
                - a32 * t29
                + a33 * t27
            )
            + b3
            * (
                -a03 * t9
                - a10 * t26
                + a13 * t1
                + a22 * t28
                - a22 * t29
                + a23 * t27
            )
        )
        num2 = (
            b0
            * (
                -a10 * t18
                + a11 * t13
                - a20 * t30
                + a20 * t7
                - a21 * t10
                + a33 * t12
            )
            - b1
            * (
                a00 * a21 * a33
                - a00 * t18
                - a01 * a20 * a33
                + a01 * t13
                + a20 * t24
                - a21 * t2
            )
            + b2
            * (
                a00 * t30
                - a00 * t7
                + a01 * t10
                + a03 * t15
                - a11 * t2
                - a33 * t0
            )
            - b3
            * (
                -a00 * t25
                + a03 * t12
                - a20 * t21
                + a20 * t22
                - a23 * t0
                + a23 * t31
            )
        )
        num3 = (
            -b0
            * (
                a10 * t19
                - a11 * t16
                + a11 * t32
                - a22 * t15
                - a30 * t6
                + a31 * t9
            )
            + b1
            * (
                -a00 * a22 * a31
                + a00 * t19
                - a01 * t16
                + a01 * t32
                - a21 * t33
                + a31 * t1
            )
            - b2
            * (
                a02 * t15
                - a11 * t33
                + a30 * t20
                - a31 * t27
                - a32 * t0
                + a32 * t31
            )
            + b3
            * (
                -a00 * t6
                + a01 * t9
                + a02 * t12
                - a11 * t1
                - a22 * t0
                + a22 * t31
            )
        )
        return jnp.stack([num0, num1, num2, num3], axis=-1) / det

    # Fallback for any size not hand-coded.
    return jnp.linalg.solve(A, b)


jax.tree_util.register_pytree_node(
    JittableDict,
    JittableDict._tree_flatten,
    JittableDict._tree_unflatten,
)
