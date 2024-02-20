from jax import config
config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp

import numpy as np

# numerical precision error
EPSILON = 1e-10

# ARRAY AND MATRIX UTILITIES

def array(els):
    return jnp.array(els, dtype='float64')

def flatten_coord(coord):
    """
    Flatten coordinates to 1d array.
    """
    return jnp.ravel(coord)

def unflatten_coord(coord_flat, spherical=True, extra_dims=[]):
    """
    Un-flatten coordinates to (n, 2, *extra_dims).
    """
    return coord_flat.reshape(-1, 2 if spherical else 3, *extra_dims)

def flatten_matrix(matrix):
    """
    Input matrix has shape (nx, ny, *block_shape).
    """
    out = np.vstack([np.hstack([block for block in row_blocks]) for row_blocks in matrix])
    return out

def unflatten_matrix(matrix, spherical=True):
    """
    Input matrix has shape (nx, ny).
    """
    dim = 2 if spherical else 3
    out = np.array([
        np.split(row_block, indices_or_sections=matrix.shape[1]//dim, axis=1)
        for row_block in np.split(matrix, indices_or_sections=matrix.shape[0]//dim, axis=0)
    ])
    return out

# POINTS

@jax.jit
def _chart(sph):
    """
    From spherical (lon, lat) coordinates to cartesian, single point.
    """
    # applies to single coordinate (1d)
    lat, lon = sph
    r = jnp.cos(lat)
    x = jnp.multiply(r, jnp.cos(lon))
    y = jnp.multiply(r, jnp.sin(lon))
    z = jnp.sin(lat)
    car_coord = array([x, y, z])
    return car_coord

@jax.jit
def sph_to_car(sph):
    """
    From spherical (lon, lat) coordinates to cartesian, array of points.
    """
    # charts work on poles as well
    sph = jnp.atleast_2d(sph)
    assert sph.ndim == 2 and sph.shape[1] == 2
    car = jax.vmap(_chart)(sph)
    return car

@jax.jit
def _inverse_chart(car):
    # single coordinate, non-pole
    lat = jnp.arcsin(car[2])
    r = jnp.sqrt(car[0]**2 + car[1]**2)
    lon = jnp.sign(car[1]) * jnp.arccos(car[0] / r)
    return array([lat, lon])

def car_to_sph(car):
    car = jnp.atleast_2d(car)
    assert car.ndim == 2 and car.shape[1] == 3
    poles = (abs(car[:, 2]) > 1 - EPSILON)
    if poles.any():
        sph = np.empty((car.shape[0], 2), dtype="float64")
        sph[~poles] = jax.vmap(_inverse_chart)(car[~poles])
        sph[car[:, 2] > 1 - EPSILON] = np.array([np.pi / 2, 0])
        sph[-car[:, 2] > 1 - EPSILON] = np.array([-np.pi / 2, 0])
    else:
        sph = jax.vmap(_inverse_chart)(car)
    # some numerical issues can lead to NaNs when y=0, remediate here
    greenwich = (abs(car[:, 1]) < EPSILON) & (~poles)
    if greenwich.any():
        sph = np.array(sph)
        sph[greenwich, 1] = np.where(car[greenwich, 0] > 0, 0., np.pi)
    return sph

# VECTORS

@jax.jit
def _d_chart(sph):
    return jax.jacfwd(_chart)(sph)

@jax.jit
def _single_sphere_tangent_basis(sph):
    basis = _d_chart(sph)
    norms = jax.vmap(jnp.linalg.norm)(basis.T)[None, :]
    return basis / norms

def sphere_tangent_basis(sph):
    sph = jnp.atleast_2d(sph)
    assert sph.ndim == 2 and sph.shape[1] == 2
    poles = np.atleast_1d(abs(jnp.sin(sph[:, 0])) > 1 - EPSILON)
    if poles.any():
        tb = np.empty((sph.shape[0], 3, 2), dtype='float64')
        tb[poles] = (np.array([[1, 0, 0], [0, 1, 0]]).T)[None, :, :]
        tb[~poles] = jax.vmap(_single_sphere_tangent_basis)(sph[~poles])
    else:
        tb = jax.vmap(_single_sphere_tangent_basis)(sph)
    return tb

@jax.jit
def _single_v_sph_to_car(sph, sph_v):
    tangent_basis = _single_sphere_tangent_basis(sph)
    return tangent_basis @ sph_v

@jax.jit
def _single_v_sph_to_car_pole(sph_v):
    tangent_basis = np.array([[1, 0, 0], [0, 1, 0]]).T
    return tangent_basis @ sph_v

def v_sph_to_car(sph, sph_v):
    """
    Vectors and their basepoints in spherical coordinates to cartesian.
        sph     basepoints in spherical coordinates
        sph_v   vectors in spherical coordinates
    """
    sph = jnp.atleast_2d(sph)
    # basepoints
    car = sph_to_car(sph)
    # vactors
    sph_v = jnp.atleast_2d(sph_v)
    assert sph_v.ndim == 2 and sph_v.shape[1] == 2
    # identify poles
    poles = (abs(jnp.sin(sph[0])) > 1 - EPSILON)
    if poles.any():
        sph, sph_v = np.array(sph), np.array(sph_v)
        # allocate space
        car_v = np.empty((sph_v.shape[0], 3), dtype="float64")
        # non-poles
        car_v[~poles] = jax.vmap(_single_v_sph_to_car)(sph[~poles], sph_v[~poles])
        # poles
        car_v[poles] = jax.vmap(_single_v_sph_to_car_pole)(sph_v[poles])
    else:
        car_v = jax.vmap(_single_v_sph_to_car)(sph, sph_v)
    return car, car_v

def v_car_to_sph(car, car_v):
    sph = car_to_sph(car)
    tangent_basis = sphere_tangent_basis(sph)
    sph_v = jax.vmap(jnp.matmul)(car_v, tangent_basis)
    return sph, sph_v