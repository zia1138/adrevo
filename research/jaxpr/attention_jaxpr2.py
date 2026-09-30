import math
import numpy as np

import jax
import jax.numpy as jnp
from jax import lax
from jax.extend import core


# ============================================================
# Attention implementations
# ============================================================

def attention_naive(q, k, v):
    """Standard exact scaled dot-product attention."""
    scale = 1.0 / jnp.sqrt(q.shape[-1])

    scores = (q @ k.T) * scale

    m = jnp.max(scores, axis=-1, keepdims=True)
    e = jnp.exp(scores - m)
    p = e / jnp.sum(e, axis=-1, keepdims=True)

    return p @ v


def make_attention_blocked(block_size):
    """
    Exact online-softmax attention.

    Never materializes the full N x N score matrix.
    """

    def attention_blocked(q, k, v):
        n, d = q.shape
        dv = v.shape[-1]

        assert n % block_size == 0

        num_blocks = n // block_size

        # Logical views of K/V as blocks.
        kb = k.reshape(num_blocks, block_size, d)
        vb = v.reshape(num_blocks, block_size, dv)

        # Running online-softmax state.
        m0 = jnp.full((n, 1), -jnp.inf, dtype=q.dtype)
        l0 = jnp.zeros((n, 1), dtype=q.dtype)
        o0 = jnp.zeros((n, dv), dtype=q.dtype)

        scale = 1.0 / jnp.sqrt(d)

        def body(state, blocks):
            m, l, o = state
            k_block, v_block = blocks

            # Only N x block_size scores exist at once.
            scores = (q @ k_block.T) * scale

            block_max = jnp.max(
                scores,
                axis=-1,
                keepdims=True,
            )

            m_new = jnp.maximum(m, block_max)

            # Rescale the old state to the new maximum.
            alpha = jnp.exp(m - m_new)

            # New block's unnormalized probabilities.
            p = jnp.exp(scores - m_new)

            l_new = (
                alpha * l
                + jnp.sum(p, axis=-1, keepdims=True)
            )

            o_new = (
                alpha * o
                + p @ v_block
            )

            # ys=None means scan does not accumulate
            # an output tensor for every iteration.
            return (m_new, l_new, o_new), None

        (_, l, o), _ = lax.scan(
            body,
            (m0, l0, o0),
            (kb, vb),
        )

        return o / l

    return attention_blocked


# ============================================================
# Recursive symbolic JAXPR memory analysis
# ============================================================

# These are treated as logical views rather than new storage.
# This is an algorithm-level model, not XLA buffer accounting.
_ALIAS_PRIMS = {
    "reshape",
    "transpose",
    "squeeze",
}


def aval_nbytes(aval):
    """Logical size of an abstract JAX value."""
    if not hasattr(aval, "shape"):
        return 0

    if not hasattr(aval, "dtype"):
        return 0

    return (
        math.prod(aval.shape)
        * np.dtype(aval.dtype).itemsize
    )


def var_nbytes(v):
    if isinstance(v, core.Var):
        return aval_nbytes(v.aval)

    return 0


def unwrap_jaxpr(j):
    if isinstance(j, core.ClosedJaxpr):
        return j.jaxpr

    return j


def scan_stacked_output_bytes(eqn):
    """
    Memory for scan `ys`.

    Carry storage is reused each iteration, while ys are
    accumulated across the full scan.
    """
    params = eqn.params

    # Current JAX representation.
    if "num_carry" in params:
        num_carry = params["num_carry"]

        return sum(
            var_nbytes(v)
            for v in eqn.outvars[num_carry:]
        )

    # Fallback: identify stacked outputs by comparing
    # body-output shape to parent-output shape.
    body = unwrap_jaxpr(params["jaxpr"])
    length = params["length"]

    total = 0

    for inner, outer in zip(
        body.outvars,
        eqn.outvars,
    ):
        if not isinstance(inner, core.Var):
            continue

        if not isinstance(outer, core.Var):
            continue

        inner_shape = tuple(inner.aval.shape)
        outer_shape = tuple(outer.aval.shape)

        if outer_shape == (length,) + inner_shape:
            total += var_nbytes(outer)

    return total


def nested_peak_bytes(eqn):
    """
    Additional peak memory occurring inside one higher-order
    primitive invocation.
    """
    name = eqn.primitive.name
    params = eqn.params

    # --------------------------------------------------------
    # cond
    #
    # Only one branch executes, so take the worst case.
    # --------------------------------------------------------
    if name == "cond":
        return max(
            (
                peak_created_bytes(branch)
                for branch in params["branches"]
            ),
            default=0,
        )

    # --------------------------------------------------------
    # while
    #
    # cond and body execute sequentially.
    # Iterations reuse storage, so do NOT multiply by
    # iteration count.
    # --------------------------------------------------------
    if name == "while":
        cond_peak = peak_created_bytes(
            params["cond_jaxpr"]
        )

        body_peak = peak_created_bytes(
            params["body_jaxpr"]
        )

        return max(cond_peak, body_peak)

    # --------------------------------------------------------
    # scan
    #
    # Body temporaries are reused each iteration.
    #
    # `ys`, however, persist because scan stacks them.
    # --------------------------------------------------------
    if name == "scan":
        body_peak = peak_created_bytes(
            params["jaxpr"]
        )

        stacked = scan_stacked_output_bytes(eqn)

        # Conservative symbolic upper bound:
        # output buffer + one body's peak.
        return body_peak + stacked

    return 0


def peak_created_bytes(
    jaxpr_like,
    exclude_final_outputs=False,
):
    """
    Compute peak logical live memory created by a JAXPR.

    Inputs and constants are assumed to already exist.

    For the top-level call, use:
        exclude_final_outputs=True

    so the result represents temporary/intermediate memory.
    """
    jaxpr = unwrap_jaxpr(jaxpr_like)

    excluded_outputs = (
        set(jaxpr.outvars)
        if exclude_final_outputs
        else set()
    )

    # Count future uses of every variable.
    remaining_uses = {}

    for eqn in jaxpr.eqns:
        for v in eqn.invars:
            if isinstance(v, core.Var):
                remaining_uses[v] = (
                    remaining_uses.get(v, 0) + 1
                )

    # Nested JAXPR outputs remain alive until return.
    if not exclude_final_outputs:
        for v in jaxpr.outvars:
            if isinstance(v, core.Var):
                remaining_uses[v] = (
                    remaining_uses.get(v, 0) + 1
                )

    live = {}

    current = 0
    peak = 0

    for eqn in jaxpr.eqns:

        # First account for memory needed while executing
        # the body of scan / while / cond.
        inner_peak = nested_peak_bytes(eqn)

        peak = max(
            peak,
            current + inner_peak,
        )

        # Equation outputs become live.
        is_alias = eqn.primitive.name in _ALIAS_PRIMS

        for v in eqn.outvars:
            if not isinstance(v, core.Var):
                continue

            if v in excluded_outputs:
                continue

            size = (
                0
                if is_alias
                else var_nbytes(v)
            )

            live[v] = size
            current += size

        peak = max(peak, current)

        # Inputs whose last use happened at this equation
        # can now be freed.
        for v in eqn.invars:
            if not isinstance(v, core.Var):
                continue

            remaining_uses[v] -= 1

            if (
                remaining_uses[v] == 0
                and v in live
            ):
                current -= live.pop(v)

    return peak


def peak_symbolic_temp_bytes(fn, *args):
    jp = jax.make_jaxpr(fn)(*args)

    return peak_created_bytes(
        jp,
        exclude_final_outputs=True,
    )


# ============================================================
# Experiment
# ============================================================

N = 512
D = 64
DV = 64
BLOCK = 64

attention_blocked = make_attention_blocked(BLOCK)


# ----------------------------
# Numerical accuracy
# ----------------------------

key = jax.random.key(0)
kq, kk, kv = jax.random.split(key, 3)

q = jax.random.normal(
    kq,
    (N, D),
    dtype=jnp.float32,
)

k = jax.random.normal(
    kk,
    (N, D),
    dtype=jnp.float32,
)

v = jax.random.normal(
    kv,
    (N, DV),
    dtype=jnp.float32,
)


reference = attention_naive(q, k, v)
candidate = attention_blocked(q, k, v)

max_abs_error = float(
    jnp.max(
        jnp.abs(reference - candidate)
    )
)

relative_l2_error = float(
    jnp.linalg.norm(reference - candidate)
    / jnp.linalg.norm(reference)
)


# ----------------------------
# Symbolic memory
# ----------------------------

q_abs = jax.ShapeDtypeStruct(
    (N, D),
    jnp.float32,
)

k_abs = jax.ShapeDtypeStruct(
    (N, D),
    jnp.float32,
)

v_abs = jax.ShapeDtypeStruct(
    (N, DV),
    jnp.float32,
)


naive_memory = peak_symbolic_temp_bytes(
    attention_naive,
    q_abs,
    k_abs,
    v_abs,
)

blocked_memory = peak_symbolic_temp_bytes(
    attention_blocked,
    q_abs,
    k_abs,
    v_abs,
)


# ----------------------------
# Results
# ----------------------------

print(f"JAX version: {jax.__version__}")
print()

print("Accuracy")
print("--------")
print(
    f"max absolute error : "
    f"{max_abs_error:.3e}"
)
print(
    f"relative L2 error  : "
    f"{relative_l2_error:.3e}"
)

print()

print("Peak symbolic temporary memory")
print("--------------------------------")
print(
    f"naive   : "
    f"{naive_memory / 2**20:.3f} MiB"
)
print(
    f"blocked : "
    f"{blocked_memory / 2**20:.3f} MiB"
)
print(
    f"reduction: "
    f"{naive_memory / blocked_memory:.2f}x"
)