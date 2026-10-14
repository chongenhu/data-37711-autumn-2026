from classification_a import binary_cross_entropy
import jax.scipy.stats as jstats

def probability(z):
    return jstats.norm.cdf(z, loc=0.0, scale=1.0)

def loss(z, y):
    return binary_cross_entropy(probability(z), y)
