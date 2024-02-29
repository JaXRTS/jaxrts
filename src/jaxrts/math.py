from jax import numpy as jnp
import jax

import numpy as np

def fermi_integral(x, n):
    
    # norm = jax.scipy.special.gamma(n+1)
    t = jnp.linspace(0, 10 * jnp.max(x), int(1E5))
    
    integrand = jnp.power(t, n) / (jnp.exp(t-x[:, jnp.newaxis]) + 1)
    
    # Test integrand
    # integrand = 1 * (t-jnp.zeros_like(x)[:, jnp.newaxis])
    
    return jax.scipy.integrate.trapezoid(integrand, t)

def numerical_inverse_fermi_integral(x, n):
    pass

def R1_mk(a, b, x):
    
    upper_max_power = jnp.size(a)
    lower_max_power = jnp.size(b)
    
    powers1 = jnp.arange(0, upper_max_power, 1)[:, jnp.newaxis]
    powers2 = jnp.arange(0, lower_max_power, 1)[:, jnp.newaxis]
    
    numerator = jnp.sum(jnp.multiply(a[:, jnp.newaxis], jnp.power(x, powers1)), axis = 0)
    denumerator = jnp.sum(jnp.multiply(b[:, jnp.newaxis], jnp.power(x, powers2)), axis = 0)
    
    return numerator / denumerator 

def inverse_fermi_12_rational_approximation(x):
    
    """
    Calculates F_1/2 using a rational function approximation as described in Antia 1993.

    """
    
    # Set parameters of the approximating functions (see Antia 1993 eq. (6)) for fixed k_1 = m_1 = 7.
    
    a = jnp.array([1.999266880833E4,
                    5.702479099336E3,
                    6.610132843877E2,
                    3.818838129486E1,
                    1.000000000000])
    
    b = jnp.array([1.771804140488E+4,
                  -2.014785161019E+3,
                   9.130355392717E+1,
                  -1.670718177489E+0])
    
    c = jnp.array([-1.277060388085E-2,
                    7.187946804945E-2,
                    -4.262314235106E-1,
                    4.997559426872E-1,
                    -1.285579118012E+0,
                    -3.930805454272E-1,
                    1.000000000000E+0])
    
    d = jnp.array([-9.745794806288E-3,
                    5.485432756838E-2,
                    -3.299466243260E-1,
                    4.077841975923E-1,
                    -1.145531476975E+0,
                    -6.067091689181E-2])
    
    return jnp.where(x < 4, jnp.log(x * R1_mk(a, b, x)),
                    x ** (2/3) * R1_mk(c, d, x ** (-2/3)))
    
    
if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    x = jnp.linspace(0, 40, 1000)
    plt.plot(x, fermi_integral(x, 0.5))
    plt.plot(x,x)
    plt.plot(x, inverse_fermi_12_rational_approximation(x))
    # plt.yscale('log')
    plt.show()