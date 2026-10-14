import jax.numpy as jnp
import jax.scipy.stats as jstats

def loss(prediction, y):
    assert prediction.ndim == y.ndim == 1 and prediction.shape == y.shape
    return -jnp.mean(jstats.t.logpdf(y - prediction, df=5.0, loc=0.0, scale=1.0))
