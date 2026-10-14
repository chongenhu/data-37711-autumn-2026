from training import CONFIG
import jax
import jax.numpy as jnp

def probability(z):
    return jax.nn.sigmoid(z)

def binary_cross_entropy(p, y):
    assert p.ndim == y.ndim == 1 and p.shape == y.shape
    p_safe = jnp.clip(p, CONFIG["probability_clip_eps"], 1 - CONFIG["probability_clip_eps"])
    return -jnp.mean(y * jnp.log(p_safe) + (1 - y) * jnp.log(1 - p_safe))

def loss(z, y):
    return binary_cross_entropy(probability(z), y)
