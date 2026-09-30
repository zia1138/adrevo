import math
import numpy as np
import jax
import jax.numpy as jnp
from jax.extend import core


# ------------------------------------------------------------
# 1. Reference attention
# ------------------------------------------------------------

def attention_naive(q, k, v):
    scale = 1.0 / jnp.sqrt(q.shape[-1])

    scores = (q @ k.T) * scale
    m = jnp.max(scores, axis=1, keepdims=True)
    e = jnp.exp(scores - m)
    p = e / jnp.sum(e, axis=1, keepdims=True)

    return p @ v


# ------------------------------------------------------------
# 2. Exact blockwise / online-softmax attention
# ------------------------------------------------------------

def make_attention_blocked(block_size):

    def attention_blocked(q, k, v):
        n, d = q.shape
        dv = v.shape[1]

        assert n % block_size == 0

        scale = 1.0 / jnp.sqrt(d)

        # Running softmax state.
        m = jnp.full((n, 1), -jnp.inf, dtype=q.dtype)
        l = jnp.zeros((n, 1), dtype=q.dtype)
        o = jnp.zeros((n, dv), dtype=q.dtype)

        for start in range(0, n, block_size):
            kb = k[start:start + block_size]
            vb = v[start:start + block_size]

            scores = (q @ kb.T) * scale

            block_max = jnp.max(scores, axis=1, keepdims=True)
            m_new = jnp.maximum(m, block_max)

            old_scale = jnp.exp(m - m_new)
            p = jnp.exp(scores - m_new)

            l = old_scale * l + jnp.sum(
                p, axis=1, keepdims=True
            )

            o = old_scale * o + p @ vb

            m = m_new

        return o / l

    return attention_blocked


# ------------------------------------------------------------
# 3. JAXPR symbolic peak-memory analysis
# ------------------------------------------------------------

def var_nbytes(v):
    aval = v.aval
    return (
        math.prod(aval.shape)
        * np.dtype(aval.dtype).itemsize
    )


def peak_symbolic_temp_bytes(closed_jaxpr):
    """
    Peak live temporary bytes in the top-level JAXPR.

    Inputs, constants, and final outputs are excluded.
    This is symbolic algorithm-level memory, not XLA/device memory.
    """

    jaxpr = getattr(closed_jaxpr, "jaxpr", closed_jaxpr)

    excluded = (
        set(jaxpr.invars)
        | set(jaxpr.constvars)
        | set(jaxpr.outvars)
    )

    # Count future uses of each variable.
    remaining_uses = {}

    for eqn in jaxpr.eqns:
        for v in eqn.invars:
            if isinstance(v, core.Var):
                remaining_uses[v] = (
                    remaining_uses.get(v, 0) + 1
                )

    live = {}
    current_bytes = 0
    peak_bytes = 0

    for eqn in jaxpr.eqns:

        # Operation produces its outputs.
        for v in eqn.outvars:
            if isinstance(v, core.Var) and v not in excluded:
                size = var_nbytes(v)
                live[v] = size
                current_bytes += size

        peak_bytes = max(peak_bytes, current_bytes)

        # Inputs whose last use just occurred can be freed.
        for v in eqn.invars:
            if not isinstance(v, core.Var):
                continue

            remaining_uses[v] -= 1

            if remaining_uses[v] == 0 and v in live:
                current_bytes -= live.pop(v)

    return peak_bytes


def symbolic_memory(fn, n, d, dv):
    q = jax.ShapeDtypeStruct((n, d), jnp.float32)
    k = jax.ShapeDtypeStruct((n, d), jnp.float32)
    v = jax.ShapeDtypeStruct((n, dv), jnp.float32)

    jp = jax.make_jaxpr(fn)(q, k, v)

    return peak_symbolic_temp_bytes(jp)


# ------------------------------------------------------------
# 4. Experiment
# ------------------------------------------------------------

N = 512
D = 64
DV = 64
BLOCK = 64

attention_blocked = make_attention_blocked(BLOCK)

key = jax.random.key(0)
k1, k2, k3 = jax.random.split(key, 3)

q = jax.random.normal(k1, (N, D))
k = jax.random.normal(k2, (N, D))
v = jax.random.normal(k3, (N, DV))


# Accuracy
ref = attention_naive(q, k, v)
test = attention_blocked(q, k, v)

max_abs_error = jnp.max(jnp.abs(ref - test))

relative_l2_error = (
    jnp.linalg.norm(ref - test)
    / jnp.linalg.norm(ref)
)


# Symbolic memory
naive_mem = symbolic_memory(
    attention_naive, N, D, DV
)

blocked_mem = symbolic_memory(
    attention_blocked, N, D, DV
)


print("Accuracy")
print("--------")
print("max abs error :", float(max_abs_error))
print("relative L2   :", float(relative_l2_error))

print()

print("Peak symbolic temporary memory")
print("--------------------------------")
print(
    "naive   :",
    naive_mem / 1024**2,
    "MiB"
)
print(
    "blocked :",
    blocked_mem / 1024**2,
    "MiB"
)
print(
    "reduction:",
    naive_mem / blocked_mem,
    "x"
)