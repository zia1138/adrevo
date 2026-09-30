import math
from collections import Counter

import jax
import jax.numpy as jnp
from jax import lax
from jax.extend import core


def newton_schulz_step(A, X):
    I = jnp.eye(A.shape[0], dtype=A.dtype)
    return X @ (2.0 * I - A @ X)


# ------------------------------------------------------------
# Primitive costs
# ------------------------------------------------------------

def numel(v):
    return math.prod(v.aval.shape)


def dot_general_ops(eqn):
    lhs_shape = eqn.invars[0].aval.shape
    out_shape = eqn.outvars[0].aval.shape

    (lhs_contract, _), _ = eqn.params["dimension_numbers"]

    k = math.prod(lhs_shape[d] for d in lhs_contract)
    outputs = math.prod(out_shape)

    return Counter({
        "matmul": 1,
        "mul": outputs * k,
        "add": outputs * max(k - 1, 0),
    })


def primitive_ops(eqn):
    name = eqn.primitive.name

    if name == "dot_general":
        return dot_general_ops(eqn)

    if name in {"add", "sub", "mul", "div", "sqrt", "exp"}:
        return Counter({
            name: numel(eqn.outvars[0])
        })

    return Counter()


# ------------------------------------------------------------
# Recursive JAXPR analysis
# ------------------------------------------------------------

def unwrap(jp):
    return jp.jaxpr if isinstance(jp, core.ClosedJaxpr) else jp


def scale_counts(counts, n):
    return Counter({
        k: v * n
        for k, v in counts.items()
    })


def max_counts(branches):
    """Componentwise worst-case upper bound."""
    out = Counter()

    for b in branches:
        for k, v in b.items():
            out[k] = max(out[k], v)

    return out


def count_jaxpr(jp):
    jp = unwrap(jp)

    total = Counter()

    for eqn in jp.eqns:
        name = eqn.primitive.name

        # ------------------------------------
        # scan: known static iteration count
        # ------------------------------------
        if name == "scan":
            body = count_jaxpr(eqn.params["jaxpr"])
            length = eqn.params["length"]

            total += scale_counts(body, length)
            continue

        # ------------------------------------
        # cond: exactly one branch executes
        # ------------------------------------
        if name == "cond":
            branches = [
                count_jaxpr(branch)
                for branch in eqn.params["branches"]
            ]

            total += max_counts(branches)
            continue

        # ------------------------------------
        # while: iteration count unknown
        # ------------------------------------
        if name == "while":
            cond_cost = count_jaxpr(
                eqn.params["cond_jaxpr"]
            )

            body_cost = count_jaxpr(
                eqn.params["body_jaxpr"]
            )

            # Report these separately because total runtime
            # depends on the number of iterations.
            for k, v in cond_cost.items():
                total[f"while_cond:{k}"] += v

            for k, v in body_cost.items():
                total[f"while_body:{k}"] += v

            continue

        total += primitive_ops(eqn)

    return total


def count_ops(fn, *args):
    jp = jax.make_jaxpr(fn)(*args)
    return count_jaxpr(jp), jp


# ------------------------------------------------------------
# Example: Newton-Schulz
# ------------------------------------------------------------

n = 128

A = jax.ShapeDtypeStruct(
    (n, n),
    jnp.float32,
)

X = jax.ShapeDtypeStruct(
    (n, n),
    jnp.float32,
)

counts, jp = count_ops(
    newton_schulz_step,
    A,
    X,
)

print(counts)
print()
print(jp)