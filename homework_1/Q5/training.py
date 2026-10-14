import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platforms", "cpu")
import jax.numpy as jnp
import optax

CONFIG = dict(seed=42, dtype=jnp.float64, learning_rate=0.01, batch_size=32,
              epochs=1000, momentum=0, weight_decay=0, probability_clip_eps=1e-12)

def predict(params, X):
    return X @ params["w"] + params["b"]

def all_finite(tree):
    return jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree_util.tree_leaves(tree)]))

def train(name, X, y, objective):
    X, y = jnp.asarray(X, dtype=CONFIG["dtype"]), jnp.asarray(y, dtype=CONFIG["dtype"])
    assert X.ndim == 2 and y.ndim == 1 and X.shape[0] == y.shape[0]
    if not bool(all_finite((X, y))):
        raise FloatingPointError(name + " has non-finite inputs or targets")
    params = {"w": jnp.zeros(X.shape[1], dtype=CONFIG["dtype"]),
              "b": jnp.array(0.0, dtype=CONFIG["dtype"])}
    optimizer = optax.sgd(learning_rate=CONFIG["learning_rate"], momentum=CONFIG["momentum"])
    state = optimizer.init(params)
    key = jax.random.PRNGKey(CONFIG["seed"])
    losses = []

    def batch_loss(params, X_batch, y_batch):
        return objective(predict(params, X_batch), y_batch)

    @jax.jit
    def step(params, state, X_batch, y_batch):
        loss, gradients = jax.value_and_grad(batch_loss)(params, X_batch, y_batch)
        updates, next_state = optimizer.update(gradients, state, params)
        next_params = optax.apply_updates(params, updates)
        valid = all_finite((predict(params, X_batch), loss, gradients, next_params))
        return next_params, next_state, valid

    @jax.jit
    def full_training_loss(params):
        prediction = predict(params, X)
        loss = objective(prediction, y)
        return loss, all_finite((prediction, loss))

    for epoch in range(CONFIG["epochs"]):
        key, shuffle_key = jax.random.split(key)
        order = jax.random.permutation(shuffle_key, len(y))
        for start in range(0, len(y), CONFIG["batch_size"]):
            indices = order[start:start + CONFIG["batch_size"]]
            params, state, valid = step(params, state, X[indices], y[indices])
            if not bool(valid):
                raise FloatingPointError(f"{name}, epoch {epoch + 1}: non-finite batch "
                                         "prediction, loss, gradient, or updated parameter")
        loss, valid = full_training_loss(params)
        if not bool(valid):
            raise FloatingPointError(name + " has non-finite full-training predictions or loss")
        losses.append(float(loss))
    return params, losses
