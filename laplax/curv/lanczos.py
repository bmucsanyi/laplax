# Copyright 2022 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modifications:
# This implementation extends the original JAX sparse linear algebra routines
# with the following features:
# - Support for mixed precision in matrix-vector products.
# - Handling of non-jittable operators using Python-level control flow.
# - Customizable eigenvalue computations using LOBPCG.
#
# The original source code can be found at:
# https://github.com/jax-ml/jax/blob/main/jax/experimental/sparse/linalg.py

"""Mixed-Precision Optional-Non-Jittable LOBPCG Wrapper for Sparse Linear Algebra.

This module provides an implementation of the Locally Optimal Block
Preconditioned Conjugate Gradient (LOBPCG) method for finding eigenvalues
and eigenvectors of large Hermitian matrices. The implementation relies on the
JAX experimental sparse linear algebra package  but extends its functionality
to support:

1. **Mixed Precision Arithmetic:**
   - Computations inside the algorithm (such as orthonormalization,
     matrix-vector products, and eigenvalue updates) can be performed
     using higher precision (e.g., `float64`), to maintain numerical
     stability in critical steps.
   - Matrix-vector products involving the operator `A` can be computed
     in higher precision (e.g., `float32`) to reduce memory
     usage and computation time.


2. **Non-Jittable Operator Support:**
   - The implementation supports `A` as a non-jittable callable, enabling
     the use of external libraries such as `scipy.sparse.linalg` for
     matrix-vector products. This is essential for cases where `A`
     cannot be expressed using JAX primitives (e.g., external libraries
     or precompiled solvers).

### Why this Wrapper?

The primary motivation for this implementation is to work around limitations
in the JAX `lax.while_loop` and sparse linear algebra primitives, which require
`A` to be jittable. By decoupling `A` from the main loop, we can support a
broader range of operators while still leveraging the performance advantages
of JAX-accelerated numerical routines where possible.
"""

import jax
import jax.numpy as jnp
from jax.experimental.sparse import linalg
from jax.experimental.sparse.linalg import (
    _check_inputs,  # noqa: PLC2701
    _eigh_ascending,  # noqa: PLC2701
    _mm,  # noqa: PLC2701
)

from laplax.types import Array, Callable, DType


def lobpcg_standard(
    A: Callable[[jax.Array], jax.Array],
    X: jax.Array,
    m: int = 100,
    tol: jax.Array | float | None = None,
    calc_dtype: DType = jnp.float64,
    a_dtype: DType = jnp.float32,
    *,
    A_jittable: bool = True,
) -> tuple[jax.Array, jax.Array, int]:
    """Compute top-k eigenvalues using LOBPCG with mixed precision.

    Args:
      A: callable representing the Hermitian matrix operation A(x).
      X: initial guess (n, k) array.
      m: max iterations
      tol: tolerance for convergence
      calc_dtype: dtype for internal calculations (float32 or float16)
      a_dtype: dtype for A calls (e.g., float64 for stable matrix-vector products)
      A_jittable: Then pass the computation to
            `jax.experimental.sparse.linalg.lobpcg_standard`.

    Returns:
        Tuple containing:
            - Eigenvalues: Array of shape (k,)
            - Eigenvectors: Array of shape (n, k)
            - Iterations: Number of iterations performed
    """
    if A_jittable:
        return linalg.lobpcg_standard(A, X, m, tol)

    n, k = X.shape
    _check_inputs(A, X)

    if tol is None:
        tol = jnp.finfo(calc_dtype).eps

    # Convert initial vectors to computation dtype
    X = X.astype(calc_dtype)

    X = _orthonormalize(X, calc_dtype=calc_dtype)
    P = _extend_basis(X, X.shape[1], calc_dtype=calc_dtype)

    # Precompute initial AX outside of jit
    # Cast to a_dtype before A and back to calc_dtype after
    AX = A(X.astype(a_dtype)).astype(calc_dtype)
    theta = jnp.sum(X * AX, axis=0, keepdims=True)
    R = AX - theta * X

    # JIT-ted iteration step that takes AX, AXPR, AS, etc. in calc_dtype
    @jax.jit
    def _iteration_first_step(X, P, R, AS):
        # Projected eigensolve
        XPR = jnp.concatenate((X, P, R), axis=1)
        theta, Q = _rayleigh_ritz_orth(AS, XPR)

        # Eigenvector X extraction
        B = Q[:, :k]
        normB = jnp.linalg.norm(B, ord=2, axis=0, keepdims=True)
        B /= normB
        X = _mm(XPR, B)
        normX = jnp.linalg.norm(X, ord=2, axis=0, keepdims=True)
        X /= normX

        # Difference terms P extraction
        q, _ = jnp.linalg.qr(Q[:k, k:].T)
        diff_rayleigh_ortho = _mm(Q[:, k:], q)
        P = _mm(XPR, diff_rayleigh_ortho)
        normP = jnp.linalg.norm(P, ord=2, axis=0, keepdims=True)
        P /= jnp.where(normP == 0, 1.0, normP)

        return X, P, R, theta

    @jax.jit
    def _iteration_second_step(X, R, theta, AX, n, tol):
        # # Compute new residuals.
        # AX = A(X)
        R = AX - theta[jnp.newaxis, :k] * X
        resid_norms = jnp.linalg.norm(R, ord=2, axis=0)

        # Compute residual norms
        reltol = jnp.linalg.norm(AX, ord=2, axis=0) + theta[:k]
        reltol *= n
        # Allow some margin for a few element-wise operations.
        reltol *= 10
        res_converged = resid_norms < tol * reltol
        converged = jnp.sum(res_converged)

        return X, R, theta[jnp.newaxis, :k], converged

    @jax.jit
    def _projection_step(X, P, R):
        R = _project_out(jnp.concatenate((X, P), axis=1), R)
        return R, jnp.concatenate((X, P, R), axis=1)

    i = 0
    converged = 0
    while i < m and converged < k:
        # Residual basis selection
        R, XPR = _projection_step(X, P, R)

        # Compute AS = AXPR = A(XPR) outside JIT at a_dtype
        AS = A(XPR.astype(a_dtype)).astype(calc_dtype)

        # Call the first iteration step
        X, P, R, theta = _iteration_first_step(X, P, R, AS)

        # Calculate AX
        AX = A(X.astype(a_dtype)).astype(calc_dtype)

        # Call the second iteration step
        X, R, theta, converged = _iteration_second_step(X, R, theta, AX, n, tol)

        i += 1

    return theta[0, :], X, i


def _orthonormalize(X: Array, calc_dtype: DType) -> Array:
    # Orthonormalize in calc_dtype
    for _ in range(2):
        X = _svqb(X, calc_dtype=calc_dtype)
    return X


def _svqb(X: Array, calc_dtype: DType) -> Array:
    X = X.astype(calc_dtype)
    norms = jnp.linalg.norm(X, ord=2, axis=0, keepdims=True)
    X /= jnp.where(norms == 0, 1.0, norms)

    inner = X.T @ X
    w, V = _eigh_ascending(inner)
    tau = jnp.finfo(inner.dtype).eps * w[0]
    padded = jnp.maximum(w, tau)
    sqrted = jnp.where(tau > 0, padded, 1.0) ** (-0.5)
    scaledV = V * sqrted[jnp.newaxis, :]
    orthoX = X @ scaledV

    keep = ((w > tau) * (jnp.diag(inner) > 0.0))[jnp.newaxis, :]
    orthoX *= keep.astype(orthoX.dtype)
    norms = jnp.linalg.norm(orthoX, ord=2, axis=0, keepdims=True)
    keep *= norms > 0.0
    orthoX /= jnp.where(keep, norms, 1.0)
    return orthoX.astype(calc_dtype)


def _extend_basis(X: Array, m: int, calc_dtype: DType) -> Array:
    n, k = X.shape
    Xupper, Xlower = jnp.split(X, [k], axis=0)
    u, s, vt = jnp.linalg.svd(Xupper)
    y = jnp.concatenate([Xupper + u @ vt, Xlower], axis=0)
    other = jnp.concatenate(
        [jnp.eye(m, dtype=calc_dtype), jnp.zeros((n - k - m, m), dtype=calc_dtype)],
        axis=0,
    )
    w = y @ (vt.T * ((2 * (1 + s)) ** (-1 / 2))[jnp.newaxis, :])
    h = -2 * jnp.linalg.multi_dot([w, w[k:, :].T, other])
    return h.at[k:].add(other).astype(calc_dtype)


def _project_out(basis: Array, U: Array) -> Array:
    """Derives component of U in the orthogonal complement of basis."""
    for _ in range(2):
        U -= _mm(basis, _mm(basis.T, U))
        U = _orthonormalize(U, U.dtype)

    # Zero out any columns that are even remotely suspicious, so the invariant that
    # that [basis, U] is zero-or-orthogonal is ensured.
    for _ in range(2):
        U -= _mm(basis, _mm(basis.T, U))
    normU = jnp.linalg.norm(U, ord=2, axis=0, keepdims=True)
    U *= (normU >= 0.99).astype(U.dtype)

    return U


def _rayleigh_ritz_orth(AS: Array, S: Array) -> tuple[Array, Array]:
    """Solve the Rayleigh-Ritz problem for `A` projected to `S`.

    This identifies `w, V` which satisfies:

    (1) `S.T A S V ~= diag(w) V`
    (2) `V` is standard orthonormal.
    """
    SAS = _mm(S.T, AS)

    # Solve the projected subsystem
    # If we could tell to eigh to stop after first k, we would.
    return _eigh_ascending(SAS)
