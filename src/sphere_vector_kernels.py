from aux_functions import *

from abc import abstractmethod
import numpy as np
from scipy.sparse import block_diag
from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.special import lpmn_values

from sklearn.gaussian_process.kernels import Kernel, Hyperparameter

# KERNEL FUNCTIONS

MAX_ELL = 35

@jax.jit
def legendre_values(x, y):
    # x, y in cartesian coordinates
    legendre_vals = lpmn_values(MAX_ELL, MAX_ELL, jnp.dot(x, y)[None], False)
    legendre_vals = jnp.squeeze(legendre_vals[0, :, :])
    return legendre_vals

# scalar manifold sphere kernel
@jax.jit
def scalar_sphere_k(x, y, kappa):
    # x, y in cartesian coordinates
    t = kappa**2 / 2
    legendre_vals = legendre_values(x, y)
    k = sum([
        jnp.exp(-t * ell * (ell + 1)) * (2 * ell + 1) * lv
        for ell, lv in enumerate(legendre_vals)
    ]) / (4 * jnp.pi)
    return k

# scalar manifold sphere kernel - Matern
# @partial(jax.jit, static_argnames=['nu'])
@jax.jit
def scalar_matern_sphere_k(x, y, kappa, nu):
    # x, y in cartesian coordinates
    t, e = 2 * nu / kappa, -nu - 1
    legendre_vals = legendre_values(x, y)
    k = sum([
        jnp.power(t + ell * (ell + 1), e) * (2 * ell + 1) * lv
        for ell, lv in enumerate(legendre_vals)
    ]) / (4 * jnp.pi)
    return k

# projected vector kernel and its gradient
@jax.jit
def projected_vector_k(x, y, kappa):
    # x, y in cartesian coordinates
    # scalar kernel
    k = scalar_sphere_k(x, y, kappa)
    # vector kernel in R3
    vk = k * jnp.eye(3)
    return vk

@jax.jit
def dlogkappa_projected_vector_k(x, y, kappa):
    # x, y in cartesian coordinates
    return kappa * jax.jacfwd(projected_vector_k, argnums=2)(x, y, kappa)

# @partial(jax.jit, static_argnames=['nu'])
@jax.jit
def projected_matern_vector_k(x, y, kappa, nu):
    # x, y in cartesian coordinates
    # scalar kernel
    k = scalar_matern_sphere_k(x, y, kappa, nu)
    # vector kernel in R3
    vk = k * jnp.eye(3)
    return vk

# @partial(jax.jit, static_argnames=['nu'])
@jax.jit
def dlogkappa_projected_matern_vector_k(x, y, kappa, nu):
    # x, y in cartesian coordinates
    return kappa * jax.jacfwd(projected_matern_vector_k, argnums=2)(x, y, kappa, nu)

# Hodge vector kernel and its gradient
@jax.jit
def hodge_vector_k(x, y, kappa):
    # x, y in cartesian coordinates
    t = kappa**2 / 2
    dd_legendre_vals = jax.jacfwd(jax.jacfwd(legendre_values, argnums=0), argnums=1)(x, y)[1:]
    # d term
    dd = jnp.multiply(
        dd_legendre_vals,
        array([
            jnp.exp(-t * ell * (ell + 1)) * (2 * ell + 1) / (4 * jnp.pi * ell * (ell + 1))
            for ell in jnp.arange(1, MAX_ELL + 1)])[:, None, None]
    ).sum(axis=0)
    # d-star term
    nx, ny = x.copy(), y.copy()
    sdsd = jnp.cross(ny, jnp.cross(nx, dd, axisa=0, axisb=1), axisa=0, axisb=0)
    # vector k
    vk = dd + sdsd
    return vk

@jax.jit
def dlogkappa_hodge_vector_k(x, y, kappa):
    # x, y in cartesian coordinates
    return kappa * jax.jacfwd(hodge_vector_k, argnums=2)(x, y, kappa)

# Hodge-Matern vector kernel and its gradient
# @partial(jax.jit, static_argnames=['nu'])
@jax.jit
def hodge_matern_vector_k(x, y, kappa, nu):
    # x, y in cartesian coordinates
    t, e = 2 * nu / kappa, -nu - 1
    dd_legendre_vals = jax.jacfwd(jax.jacfwd(legendre_values, argnums=0), argnums=1)(x, y)[1:]
    # d term
    dd = jnp.multiply(
        dd_legendre_vals,
        array([
            jnp.power(t + ell * (ell + 1), e) * (2 * ell + 1) / (4 * jnp.pi * ell * (ell + 1))
            for ell in jnp.arange(1, MAX_ELL + 1)])[:, None, None]
    ).sum(axis=0)
    # d-star term
    nx, ny = x.copy(), y.copy()
    sdsd = jnp.cross(ny, jnp.cross(nx, dd, axisa=0, axisb=1), axisa=0, axisb=0)
    # vector k
    vk = dd + sdsd
    return vk

# @partial(jax.jit, static_argnames=['nu'])
@jax.jit
def dlogkappa_hodge_matern_vector_k(x, y, kappa, nu):
    # x, y in cartesian coordinates
    return kappa * jax.jacfwd(hodge_matern_vector_k, argnums=2)(x, y, kappa, nu)

# div and curl free kernels
@jax.jit
def hodge_vector_div_free_k(x, y, kappa):
    # x, y in cartesian coordinates
    t = kappa**2 / 2
    dd_legendre_vals = jax.jacfwd(jax.jacfwd(legendre_values, argnums=0), argnums=1)(x, y)[1:]
    # d term
    dd = jnp.multiply(
        dd_legendre_vals,
        array([
            jnp.exp(-t * ell * (ell + 1)) * (2 * ell + 1) / (4 * jnp.pi * ell * (ell + 1))
            for ell in jnp.arange(1, MAX_ELL + 1)])[:, None, None]
    ).sum(axis=0)
    # d-star term
    nx, ny = x.copy(), y.copy()
    sdsd = jnp.cross(ny, jnp.cross(nx, dd, axisa=0, axisb=1), axisa=0, axisb=0)
    # vector k
    vk = sdsd
    return vk

@jax.jit
def hodge_vector_curl_free_k(x, y, kappa):
    # x, y in cartesian coordinates
    t = kappa**2 / 2
    dd_legendre_vals = jax.jacfwd(jax.jacfwd(legendre_values, argnums=0), argnums=1)(x, y)[1:]
    # d term
    dd = jnp.multiply(
        dd_legendre_vals,
        array([
            jnp.exp(-t * ell * (ell + 1)) * (2 * ell + 1) / (4 * jnp.pi * ell * (ell + 1))
            for ell in jnp.arange(1, MAX_ELL + 1)])[:, None, None]
    ).sum(axis=0)
    # vector k
    vk = dd
    return vk

@jax.jit
def dlogkappa_hodge_vector_div_free_k(x, y, kappa):
    # x, y in cartesian coordinates
    return kappa * jax.jacfwd(hodge_vector_div_free_k, argnums=2)(x, y, kappa)

@jax.jit
def dlogkappa_hodge_vector_curl_free_k(x, y, kappa):
    # x, y in cartesian coordinates
    return kappa * jax.jacfwd(hodge_vector_curl_free_k, argnums=2)(x, y, kappa)


@jax.jit
def hodge_matern_div_free_vector_k(x, y, kappa, nu):
    # x, y in cartesian coordinates
    t, e = 2 * nu / kappa, -nu - 1
    dd_legendre_vals = jax.jacfwd(jax.jacfwd(legendre_values, argnums=0), argnums=1)(x, y)[1:]
    # d term
    dd = jnp.multiply(
        dd_legendre_vals,
        array([
            jnp.power(t + ell * (ell + 1), e) * (2 * ell + 1) / (4 * jnp.pi * ell * (ell + 1))
            for ell in jnp.arange(1, MAX_ELL + 1)])[:, None, None]
    ).sum(axis=0)
    # d-star term
    nx, ny = x.copy(), y.copy()
    sdsd = jnp.cross(ny, jnp.cross(nx, dd, axisa=0, axisb=1), axisa=0, axisb=0)
    # vector k
    vk = sdsd
    return vk

@jax.jit
def dlogkappa_hodge_matern_div_free_vector_k(x, y, kappa, nu):
    # x, y in cartesian coordinates
    return kappa * jax.jacfwd(hodge_matern_div_free_vector_k, argnums=2)(x, y, kappa, nu)


@jax.jit
def hodge_matern_curl_free_vector_k(x, y, kappa, nu):
    # x, y in cartesian coordinates
    t, e = 2 * nu / kappa, -nu - 1
    dd_legendre_vals = jax.jacfwd(jax.jacfwd(legendre_values, argnums=0), argnums=1)(x, y)[1:]
    # d term
    dd = jnp.multiply(
        dd_legendre_vals,
        array([
            jnp.power(t + ell * (ell + 1), e) * (2 * ell + 1) / (4 * jnp.pi * ell * (ell + 1))
            for ell in jnp.arange(1, MAX_ELL + 1)])[:, None, None]
    ).sum(axis=0)
    # vector k
    vk = dd
    return vk

@jax.jit
def dlogkappa_hodge_matern_curl_free_vector_k(x, y, kappa, nu):
    # x, y in cartesian coordinates
    return kappa * jax.jacfwd(hodge_matern_div_free_vector_k, argnums=2)(x, y, kappa, nu)

# KERNEL CLASSES

class VectorKernel(Kernel):
    """
    Generic vector kernel.
    All inputs are in spherical coordinates.
    Note that the kernels will not have normalized variance a priori.
    """
    
    def __call__(self, X, Y=None, eval_gradient=False):
        X = unflatten_coord(X)
        tangent_basis_X = sphere_tangent_basis(X)
        X = sph_to_car(X)
        if Y is None:
            Y = X
            tangent_basis_Y = tangent_basis_X
        else:
            Y = unflatten_coord(Y)
            tangent_basis_Y = sphere_tangent_basis(Y)
            Y = sph_to_car(Y)
            if eval_gradient:
                raise ValueError("Gradient can only be evaluated when Y is None.")

        K = self.compute_kernel(X, tangent_basis_X, Y, tangent_basis_Y)
        
        if eval_gradient:
            K_gradient = self.compute_kernel_gradient(
                X, tangent_basis_X, Y, tangent_basis_Y
            )
            return K, K_gradient
        return K
    
    def compute_kernel(self, X, tangent_basis_X, Y, tangent_basis_Y):
        K = flatten_matrix(array([
            jax.vmap(
                self.single_kernel_call, in_axes=(None, None, 0, 0)
            )(x, tb_x, Y, tangent_basis_Y)
            # partially parallelized for speed
            for x, tb_x in zip(X, tangent_basis_X)
        ]))
        return K
    
    def compute_kernel_gradient(self, X, tangent_basis_X, Y, tangent_basis_Y):
        K = flatten_matrix(array([
            jax.vmap(
                self.single_kernel_gradient_call, in_axes=(None, None, 0, 0)
            )(
                x, tb_x, Y, tangent_basis_Y
            )
            for x, tb_x in zip(X, tangent_basis_X)
        ]))
        return K
    
    def is_stationary(self):
        return False
    
    def diag(self, X):
        raise notImplementedError()
    
    @abstractmethod
    def single_kernel_call(self, x, tb_x, y, tb_y):
        # this takes cartesian coordiantes as inputs
        # and returns matrix in spherical coordinates
        pass
    
    @abstractmethod
    def single_kernel_gradient_call(self, x, tb_x, y, tb_y):
        # this takes cartesian coordiantes as inputs
        # and returns matrix in spherical coordinates
        pass


class ProjectedSphereKernel(VectorKernel):
    
    def __init__(self, kappa=1.0, kappa_bounds=(.05, 1e5)):
        super().__init__()
        
        self.kappa = kappa
        self.kappa_bounds = kappa_bounds
    
    @property
    def hyperparameter_kappa(self):
        return Hyperparameter("kappa", "numeric", self.kappa_bounds)
    
    def single_kernel_call(self, x, tb_x, y, tb_y):
        return tb_x.T @ projected_vector_k(x, y, self.kappa) @ tb_y
    
    def single_kernel_gradient_call(self, x, tb_x, y, tb_y):
        return (tb_x.T @ dlogkappa_projected_vector_k(x, y, self.kappa) @ tb_y)[:, :, None]
    
    def __repr__(self):
        return "{0}(kappa={1:.3g})".format(self.__class__.__name__, self.kappa)
    
    def compute_kernel_gradient(self, X, tangent_basis_X, Y, tangent_basis_Y):
        if self.hyperparameter_kappa.fixed:
            return np.empty((2 * X.shape[0], 2 * Y.shape[0], 0))
        return super().compute_kernel_gradient(X, tangent_basis_X, Y, tangent_basis_Y)


class ProjectedMaternSphereKernel(VectorKernel):
    
    def __init__(self, kappa=1.0, kappa_bounds=(.0025, 1e5), nu=1.5):
        super().__init__()
        
        self.kappa = kappa
        self.kappa_bounds = kappa_bounds
        
        # nu is not an hyperparameter!
        self.nu = nu
    
    @property
    def hyperparameter_kappa(self):
        return Hyperparameter("kappa", "numeric", self.kappa_bounds)
    
    def single_kernel_call(self, x, tb_x, y, tb_y):
        return tb_x.T @ projected_matern_vector_k(x, y, self.kappa, self.nu) @ tb_y
    
    def single_kernel_gradient_call(self, x, tb_x, y, tb_y):
        return (tb_x.T @ dlogkappa_projected_matern_vector_k(x, y, self.kappa, self.nu) @ tb_y)[:, :, None]
    
    def __repr__(self):
        return "{0}(kappa={1:.3g}, nu={2:.3g})".format(self.__class__.__name__, self.kappa, self.nu)
    
    def compute_kernel_gradient(self, X, tangent_basis_X, Y, tangent_basis_Y):
        if self.hyperparameter_kappa.fixed:
            return np.empty((2 * X.shape[0], 2 * Y.shape[0], 0))
        return super().compute_kernel_gradient(X, tangent_basis_X, Y, tangent_basis_Y)


class HodgeSphereKernel(VectorKernel):
    
    def __init__(self, kappa=1.0, kappa_bounds=(.03, 1e5)):
        super().__init__()
        
        self.kappa = kappa
        self.kappa_bounds = kappa_bounds
    
    @property
    def hyperparameter_kappa(self):
        return Hyperparameter("kappa", "numeric", self.kappa_bounds)
    
    def single_kernel_call(self, x, tb_x, y, tb_y):
        return tb_x.T @ hodge_vector_k(x, y, self.kappa) @ tb_y
    
    def single_kernel_gradient_call(self, x, tb_x, y, tb_y):
        return (tb_x.T @ dlogkappa_hodge_vector_k(x, y, self.kappa) @ tb_y)[:, :, None]
    
    def __repr__(self):
        return "{0}(kappa={1:.3g})".format(self.__class__.__name__, self.kappa)
    
    def compute_kernel_gradient(self, X, tangent_basis_X, Y, tangent_basis_Y):
        if self.hyperparameter_kappa.fixed:
            return np.empty((2 * X.shape[0], 2 * Y.shape[0], 0))
        return super().compute_kernel_gradient(X, tangent_basis_X, Y, tangent_basis_Y)


class HodgeMaternSphereKernel(VectorKernel):
    
    def __init__(self, kappa=1.0, kappa_bounds=(.001, 1e5), nu=1.5):
        super().__init__()
        
        self.kappa = kappa
        self.kappa_bounds = kappa_bounds
        
        # nu is not an hyperparameter!
        self.nu = nu
    
    @property
    def hyperparameter_kappa(self):
        return Hyperparameter("kappa", "numeric", self.kappa_bounds)
    
    def single_kernel_call(self, x, tb_x, y, tb_y):
        return tb_x.T @ hodge_matern_vector_k(x, y, self.kappa, self.nu) @ tb_y
    
    def single_kernel_gradient_call(self, x, tb_x, y, tb_y):
        return (tb_x.T @ dlogkappa_hodge_matern_vector_k(x, y, self.kappa, self.nu) @ tb_y)[:, :, None]
    
    def __repr__(self):
        return "{0}(kappa={1:.3g}, nu={2:.3g})".format(self.__class__.__name__, self.kappa, self.nu)
    
    def compute_kernel_gradient(self, X, tangent_basis_X, Y, tangent_basis_Y):
        if self.hyperparameter_kappa.fixed:
            return np.empty((2 * X.shape[0], 2 * Y.shape[0], 0))
        return super().compute_kernel_gradient(X, tangent_basis_X, Y, tangent_basis_Y)


class HodgeDivFreeSphereKernel(VectorKernel):
    
    def __init__(self, kappa=1.0, kappa_bounds=(.03, 1e5)):
        super().__init__()
        
        self.kappa = kappa
        self.kappa_bounds = kappa_bounds
    
    @property
    def hyperparameter_kappa(self):
        return Hyperparameter("kappa", "numeric", self.kappa_bounds)
    
    def single_kernel_call(self, x, tb_x, y, tb_y):
        return tb_x.T @ hodge_vector_div_free_k(x, y, self.kappa) @ tb_y
    
    def single_kernel_gradient_call(self, x, tb_x, y, tb_y):
        return (tb_x.T @ dlogkappa_hodge_vector_div_free_k(x, y, self.kappa) @ tb_y)[:, :, None]
    
    def __repr__(self):
        return "{0}(kappa={1:.3g})".format(self.__class__.__name__, self.kappa)
    
    def compute_kernel_gradient(self, X, tangent_basis_X, Y, tangent_basis_Y):
        if self.hyperparameter_kappa.fixed:
            return np.empty((2 * X.shape[0], 2 * Y.shape[0], 0))
        return super().compute_kernel_gradient(X, tangent_basis_X, Y, tangent_basis_Y)


class HodgeCurlFreeSphereKernel(VectorKernel):
    
    def __init__(self, kappa=1.0, kappa_bounds=(.03, 1e5)):
        super().__init__()
        
        self.kappa = kappa
        self.kappa_bounds = kappa_bounds
    
    @property
    def hyperparameter_kappa(self):
        return Hyperparameter("kappa", "numeric", self.kappa_bounds)
    
    def single_kernel_call(self, x, tb_x, y, tb_y):
        return tb_x.T @ hodge_vector_curl_free_k(x, y, self.kappa) @ tb_y
    
    def single_kernel_gradient_call(self, x, tb_x, y, tb_y):
        return (tb_x.T @ dlogkappa_hodge_vector_curl_free_k(x, y, self.kappa) @ tb_y)[:, :, None]
    
    def __repr__(self):
        return "{0}(kappa={1:.3g})".format(self.__class__.__name__, self.kappa)
    
    def compute_kernel_gradient(self, X, tangent_basis_X, Y, tangent_basis_Y):
        if self.hyperparameter_kappa.fixed:
            return np.empty((2 * X.shape[0], 2 * Y.shape[0], 0))
        return super().compute_kernel_gradient(X, tangent_basis_X, Y, tangent_basis_Y)


class HodgeMaternDivFreeSphereKernel(VectorKernel):
    
    def __init__(self, kappa=1.0, kappa_bounds=(.001, 1e5), nu=1.5):
        super().__init__()
        
        self.kappa = kappa
        self.kappa_bounds = kappa_bounds
        
        # nu is not an hyperparameter!
        self.nu = nu
    
    @property
    def hyperparameter_kappa(self):
        return Hyperparameter("kappa", "numeric", self.kappa_bounds)
    
    def single_kernel_call(self, x, tb_x, y, tb_y):
        return tb_x.T @ hodge_matern_div_free_vector_k(x, y, self.kappa, self.nu) @ tb_y
    
    def single_kernel_gradient_call(self, x, tb_x, y, tb_y):
        return (tb_x.T @ dlogkappa_hodge_matern_div_free_vector_k(x, y, self.kappa, self.nu) @ tb_y)[:, :, None]
    
    def __repr__(self):
        return "{0}(kappa={1:.3g}, nu={2:.3g})".format(self.__class__.__name__, self.kappa, self.nu)
    
    def compute_kernel_gradient(self, X, tangent_basis_X, Y, tangent_basis_Y):
        if self.hyperparameter_kappa.fixed:
            return np.empty((2 * X.shape[0], 2 * Y.shape[0], 0))
        return super().compute_kernel_gradient(X, tangent_basis_X, Y, tangent_basis_Y)


class HodgeMaternCurlFreeSphereKernel(VectorKernel):
    
    def __init__(self, kappa=1.0, kappa_bounds=(.001, 1e5), nu=1.5):
        super().__init__()
        
        self.kappa = kappa
        self.kappa_bounds = kappa_bounds
        
        # nu is not an hyperparameter!
        self.nu = nu
    
    @property
    def hyperparameter_kappa(self):
        return Hyperparameter("kappa", "numeric", self.kappa_bounds)
    
    def single_kernel_call(self, x, tb_x, y, tb_y):
        return tb_x.T @ hodge_matern_curl_free_vector_k(x, y, self.kappa, self.nu) @ tb_y
    
    def single_kernel_gradient_call(self, x, tb_x, y, tb_y):
        return (tb_x.T @ dlogkappa_hodge_matern_curl_free_vector_k(x, y, self.kappa, self.nu) @ tb_y)[:, :, None]
    
    def __repr__(self):
        return "{0}(kappa={1:.3g}, nu={2:.3g})".format(self.__class__.__name__, self.kappa, self.nu)
    
    def compute_kernel_gradient(self, X, tangent_basis_X, Y, tangent_basis_Y):
        if self.hyperparameter_kappa.fixed:
            return np.empty((2 * X.shape[0], 2 * Y.shape[0], 0))
        return super().compute_kernel_gradient(X, tangent_basis_X, Y, tangent_basis_Y)
