import numba
import numpy as np
from joblib import Parallel, delayed

from . import utils
from .CPAlgorithm import CPAlgorithm


class BE(CPAlgorithm):
    """Borgatti Everett algorithm.

    Algorithm for finding single core-periphery pair in networks.

    S. P. Borgatti and M. G. Everett. Models of core/periphery structures. Social Networks, 21, 375–395, 2000

    .. highlight:: python
    .. code-block:: python

        >>> import cpnet
        >>> be = cpnet.BE()
        >>> be.detect(G)
        >>> pair_id = be.get_pair_id()
        >>> coreness = be.get_coreness()

    .. note::

        - [ ] weighted
        - [ ] directed
        - [ ] multiple groups of core-periphery pairs
        - [ ] continuous core-periphery structure
    """

    def __init__(self, num_runs=10):
        """Initialize algorithm.

        :param num_runs: number of runs, defaults to 10
        :type num_runs: int, optional
        """
        self.num_runs = num_runs
        self.n_jobs = 1

    def detect(self, G):
        """Detect core-periphery structure.

        :param G: Graph
        :type G: networkx.Graph or scipy sparse matrix
        :return: None
        :rtype: None
        """

        A, nodelabel = utils.to_adjacency_matrix(G)

        def _detect(A_indptr, A_indices, A_data, num_nodes):
            x = _kernighan_lin_(A_indptr, A_indices, A_data, num_nodes)
            x = x.astype(int)
            cids = np.zeros(num_nodes).astype(int)
            Q, qs = _score_(A_indptr, A_indices, A_data, cids, x, num_nodes)
            return {"cids": cids, "x": x, "q": Q}

        res = Parallel(n_jobs=self.n_jobs)(
            delayed(_detect)(A.indptr, A.indices, A.data, A.shape[0])
            for i in range(self.num_runs)
        )
        res = max(res, key=lambda x: x["q"])
        cids, x, Q = res["cids"], res["x"], res["q"]
        self.nodelabel = nodelabel
        self.c_ = cids.astype(int)
        self.x_ = x.astype(int)
        self.Q_ = Q
        self.qs_ = [Q]

    def _score(self, A, c, x):
        """Calculate the strength of core-periphery pairs.

        :param A: Adjacency amtrix
        :type A: scipy sparse matrix
        :param c: group to which a node belongs
        :type c: dict
        :param x: core (x=1) or periphery (x=0)
        :type x: dict
        :return: strength of core-periphery
        :rtype: float
        """
        num_nodes = A.shape[0]
        Q, qs = _score_(A.indptr, A.indices, A.data, c, x, num_nodes)
        return qs

@numba.njit(cache=True)
def _kernighan_lin_(A_indptr, A_indices, A_data, num_nodes):
    """
    Robust greedy ascent for the BE (binary) objective:
      - At each step, try flipping each node's label (0↔1),
      - pick the flip that yields the highest Q,
      - apply it if it improves Q by > tol,
      - stop when no single flip improves Q.

    Assumes:
      - simple, undirected, binary CSR (upper/lower both present; we divide by 2 where needed)
      - x in {0,1} (1=core, 0=periphery)
    """

    def compute_M(num_nodes, A_indptr):
        degsum = 0.0
        for i in range(num_nodes):
            degsum += (A_indptr[i+1] - A_indptr[i])
        return 0.5 * degsum  # undirected: each edge counted twice

    def compute_Q_for_x(x, num_nodes, A_indptr, A_indices, M, T, pa):
        # compute m_core_or (numer) and nc from scratch
        numer2 = 0.0  # counted twice (i->j and j->i)
        nc = 0.0
        for i in range(num_nodes):
            xi = x[i]
            nc += xi
            start, end = A_indptr[i], A_indptr[i+1]
            for t in range(start, end):
                j = A_indices[t]
                xj = x[j]
                numer2 += (xi + xj - xi * xj)
        numer = 0.5 * numer2  # undirected

        # M_b = number of node-pairs with ≥1 core endpoint
        # closed form: M_b = (nc*(nc-1) + 2*nc*(N-nc)) / 2
        N = float(num_nodes)
        Mb = (nc * (nc - 1.0) + 2.0 * nc * (N - nc)) * 0.5

        # pb = Mb / T, with clipping
        pb = Mb / T
        eps = 1e-12
        if pb < eps:
            pb = eps
        elif pb > 1.0 - eps:
            pb = 1.0 - eps

        denom = (np.sqrt(pa * (1.0 - pa)) * np.sqrt(pb * (1.0 - pb)))
        return (numer - pa * Mb) / denom

    # --- precompute graph constants ---
    T = 0.5 * num_nodes * (num_nodes - 1.0)
    if T <= 0.0:
        # empty or single-node graph
        return np.zeros(num_nodes, dtype=np.float64)

    M = compute_M(num_nodes, A_indptr)
    pa = M / T
    eps = 1e-12
    if pa < eps:
        pa = eps
    elif pa > 1.0 - eps:
        pa = 1.0 - eps

    x = np.zeros(num_nodes, dtype=np.float64)
    for i in range(num_nodes):
        x[i] = 1.0 if np.random.rand() < 0.5 else 0.0

    Q_curr = compute_Q_for_x(x, num_nodes, A_indptr, A_indices, M, T, pa)

    tol = 1e-12
    max_outer = 5 * num_nodes  # safety cap

    for _ in range(max_outer):
        best_Q = Q_curr
        best_k = -1

        # try flipping each node and evaluate exact Q
        for k in range(num_nodes):
            x[k] = 1.0 - x[k]  # flip
            Q_try = compute_Q_for_x(x, num_nodes, A_indptr, A_indices, M, T, pa)
            if Q_try > best_Q + tol:
                best_Q = Q_try
                best_k = k
            x[k] = 1.0 - x[k]  # revert

        if best_k == -1:
            break

        x[best_k] = 1.0 - x[best_k]
        Q_curr = best_Q

    return x



@numba.jit(nopython=True, cache=True)
def _score_(A_indptr, A_indices, A_data, _c, _x, num_nodes):

    M = 0.0
    pa = 0
    pb = 0
    nc = 0
    mcc = 0
    for i in range(num_nodes):
        nc += _x[i]
        neighbors = A_indices[A_indptr[i] : A_indptr[i + 1]]
        for  j in neighbors:
            mcc += _x[i] + _x[j] - _x[i] * _x[j]
            M += 1

    mcc = mcc / 2
    M = M / 2
    M_b = (nc * (nc - 1) + 2 * nc * (num_nodes - nc)) / 2
    pa = M / np.maximum(1, num_nodes * (num_nodes - 1) / 2)
    pb = M_b / np.maximum(1, num_nodes * (num_nodes - 1) / 2)

    Q = (mcc - pa * M_b) / np.maximum(
        1e-20, (np.sqrt(pa * (1 - pa)) * np.sqrt(pb * (1 - pb)))
    )
    #Q = Q / np.maximum(1, (num_nodes * (num_nodes - 1) / 2))

    return Q, [Q]
