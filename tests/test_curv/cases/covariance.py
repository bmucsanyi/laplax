import jax
import jax.numpy as jnp
import pytest_cases


def draw_random_onb(key, shape: tuple[int]) -> jnp.ndarray:
    """Draw a random orthogonal matrix and a random vector of singular values.

    Note: Shape (n, m) with n being the full rank and m being the low rank.
    """
    keys = jax.random.split(key, 2)
    A = jax.random.uniform(keys[0], 2 * shape[:1])
    A = 0.5 * (A + A.T)
    A += jnp.eye(shape[0])
    U, _ = jnp.linalg.qr(A)
    S = jax.random.uniform(keys[1], (shape[1],))
    return U[: shape[1], :].T, S


class CurvatureTask:
    method = None

    def __init__(self, seed, size=100, rank=10):
        # Set seed and keys
        self.seed = seed
        key = jax.random.key(seed)
        keys = jax.random.split(key, 3)
        self.key_arr = keys[0]
        self.key_curv_est = keys[1]

        # Initialize task
        self.size = size
        self.rank = rank
        self._initialize()

    def _initialize(self):
        # Get low rank array (repr. GGN and Hessian)
        u, s = draw_random_onb(self.key_arr, (self.size, self.rank))
        self.arr = u @ jnp.diag(s) @ u.T
        self.u = u
        self.s = s

    def adjust_curv_est(self):
        raise NotImplementedError

    def adjust_prec(self):
        raise NotImplementedError

    @property
    def true_curv(self):
        raise NotImplementedError

    @property
    def arr_mv(self):
        def arr_mv(vec):
            return self.arr @ vec

        return arr_mv

    @property
    def tree_like(self):
        return jax.numpy.ones((100,))


class LowRankCurvatureTask(CurvatureTask):
    method = "low_rank"

    def adjust_curv_est(self, low_rank_terms: dict):
        U = low_rank_terms["U"]
        S = low_rank_terms["S"]
        return U @ jnp.diag(S) @ U.T

    def adjust_prec(self, low_rank_terms: dict):
        U = low_rank_terms["U"]
        S = low_rank_terms["S"]
        scalar = low_rank_terms["scalar"]
        return U @ jnp.diag(S) @ U.T + scalar * jnp.eye(self.size)

    @property
    def true_curv(self):
        return self.arr


class DiagonalCurvatureTask(CurvatureTask):
    method = "diagonal"

    def adjust_curv_est(self, curv_est):
        return jnp.diag(curv_est)

    def adjust_prec(self, prec):
        return jnp.diag(prec)

    @property
    def true_curv(self):
        return jnp.diag(jnp.diag(self.arr))


class FullCurvatureTask(CurvatureTask):
    method = "full"

    def adjust_curv_est(self, curv_est):
        return curv_est

    def adjust_prec(self, prec):
        return prec

    @property
    def true_curv(self):
        return self.arr


@pytest_cases.parametrize(
    "task_class", [LowRankCurvatureTask, DiagonalCurvatureTask, FullCurvatureTask]
)
@pytest_cases.parametrize("seed", [0, 1])
def case_posterior_covariance(task_class, seed):
    return task_class(seed=seed)