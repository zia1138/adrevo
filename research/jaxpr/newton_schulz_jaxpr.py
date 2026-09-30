import math
import jax
import jax.numpy as jnp


def newton_schulz_step(A, X):
    I = jnp.eye(A.shape[0], dtype=A.dtype)
    return X @ (2.0 * I - A @ X)


def dot_general_ops(eqn):
    lhs_shape = eqn.invars[0].aval.shape
    out_shape = eqn.outvars[0].aval.shape

    (lhs_contract, _), _ = eqn.params["dimension_numbers"]

    k = math.prod(lhs_shape[d] for d in lhs_contract)
    outputs = math.prod(out_shape)

    return {
        "mul": outputs * k,
        "add": outputs * max(k - 1, 0),
    }


def count_ops(fn, *args):
    jp = jax.make_jaxpr(fn)(*args)

    counts = {
        "matmul": 0,
        "mul": 0,
        "add": 0,
        "sub": 0,
    }

    for eqn in jp.jaxpr.eqns:
        name = eqn.primitive.name

        if name == "dot_general":
            c = dot_general_ops(eqn)
            counts["matmul"] += 1
            counts["mul"] += c["mul"]
            counts["add"] += c["add"]

        elif name in {"mul", "add", "sub"}:
            n = math.prod(eqn.outvars[0].aval.shape)
            counts[name] += n

    return counts, jp


n = 128

A = jax.ShapeDtypeStruct((n, n), jnp.float32)
X = jax.ShapeDtypeStruct((n, n), jnp.float32)

counts, jp = count_ops(newton_schulz_step, A, X)

print(counts)
print()
print(jp)