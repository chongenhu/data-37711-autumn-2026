import jax.numpy as jnp

def loss(prediction, y):
    assert prediction.ndim == y.ndim == 1 and prediction.shape == y.shape
    return jnp.mean((y - prediction) ** 2)
