"""
This submodule is dedicated to calculate the contribution of tightly bound electrons to the dynamic structure factor.
"""

from .units import ureg, Quantity
from typing import List

import jax
from jax import jit
import jax.numpy as jnp
import numpy as onp

import logging
logger = logging.getLogger(__name__)

import jpu

