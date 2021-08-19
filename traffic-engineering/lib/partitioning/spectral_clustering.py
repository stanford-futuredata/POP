from .abstract_partitioning_method import AbstractPartitioningMethod
from sklearn.cluster import KMeans
from .utils import all_partitions_contiguous
import numpy as np
import networkx as nx
import time


# Run NJW spectral clustering, use eigengap heuristic to select the number of partitions
class SpectralClustering(AbstractPartitioningMethod):
    def __init__(self, num_partitions=None, weighted=True, seed=0):
        super().__init__(num_partitions=num_partitions, weighted=weighted)
        if weighted:
            self._adj_mat = lambda G: np.asarray(
                nx.adjacency_matrix(G, weight="capacity").todense(), dtype=np.float64
            )
        else:
            self._adj_mat = lambda G: np.asarray(
                nx.adjacency_matrix(G, weight="").todense(), dtype=np.float64
            )
        self.seed = seed

    @property
    def name(self):
        return "spectral_clustering"

    def run_k_means_on_eigenvectors(self, eigvecs, num_nodes):
        start = time.time()
        V = eigvecs[:, : self._num_partitions]
        U = V / np.linalg.norm(V, axis=1).reshape(num_nodes, 1)

        k_means = KMeans(self._num_partitions, n_init=100, random_state=self.seed).fit(
            U
        )
        self.runtime = time.time() - start
        return k_means.labels_

    # Normalized spectral clustering according to Ng, Jordan, and Weiss (2002)
    def _partition_impl(self, problem):
        def is_symmetric(a, rtol=1e-05, atol=1e-08):
            return np.allclose(a, a.T, rtol=rtol, atol=atol)

        def is_pos_semi_def(x):
            return np.all(np.linalg.eigvals(x) >= -1e-5)

        G = problem.G.copy()
        num_nodes = len(G.nodes)
        W = self._adj_mat(G.to_undirected())

        # 1) Build Laplacian matrix L of the graph
        # = I − D−1/2W D−1/2, where D−1/2 is a diagonal matrix with (D−1/2)ii = (Dii)−1/2
        D = np.diag(np.sum(W, axis=1))
        D_norm = np.power(D, -0.5)
        D_norm[D_norm == np.inf] = 0.0
        L = np.identity(W.shape[0]) - D_norm.dot(W).dot(D_norm)
        assert is_symmetric(L)
        assert is_pos_semi_def(L)

        # 2) Find eigenvalues and eigenvalues of L
        eigvals, eigvecs = np.linalg.eig(L)
        eigvals, eigvecs = eigvals.astype(np.float32), eigvecs.astype(np.float32)
        eigvecs = eigvecs[:, np.argsort(eigvals)]
        eigvals = eigvals[np.argsort(eigvals)]
        self.eigenvals = eigvals

        # 3) If number of partitions was not set, find largest eigengap between eigenvalues. If resulting
        #    partition is not contiguous, try the 2nd-largest eigengap, and so on...
        if not hasattr(self, "_num_partitions"):
            max_num_parts = int(num_nodes / 4)
            print(
                "Using eigengap heuristic to select number of partitions, max: {}".format(
                    max_num_parts
                )
            )
            self.eigengaps = np.array(
                [
                    eigvals[i + 1] - eigvals[i]
                    for i in range(len(eigvals[:max_num_parts]) - 1)
                ]
            )

            k = 0
            indices = self.eigengaps.argsort()[::-1]

            while k < len(indices):
                self._num_partitions = indices[k]
                print("Trying {} partitions".format(self._num_partitions))
                p_v = self.run_k_means_on_eigenvectors(eigvecs, num_nodes)
                if all_partitions_contiguous(problem, p_v):
                    break
                k += 1
            if k == len(indices):
                raise Exception("could not find valid partitioning")

            print(
                "Eigengap heuristic selected {} partitions".format(self._num_partitions)
            )
            return p_v

        else:
            return self.run_k_means_on_eigenvectors(eigvecs, num_nodes)
