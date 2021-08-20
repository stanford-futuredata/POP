from .utils import uni_rand
from collections import defaultdict

import networkx as nx
import numpy as np

import pickle
import os


class TrafficMatrix(object):
    def __init__(self, problem, tm, seed, scale_factor):
        debug = False
        self._problem = problem
        self._seed = seed
        self._scale_factor = scale_factor
        if tm is not None:
            self._tm = tm if isinstance(tm, np.ndarray) else np.array(tm)
            assert self._tm.shape[0] == self._tm.shape[1]  # Must be square
        else:
            self._init_traffic_matrix()
            self._tm *= self.scale_factor

        if debug:
            print("% full:", np.count_nonzero(self._tm) / self._tm.size)
            print("total demand:", np.sum(self._tm))

    @property
    def problem(self):
        return self._problem

    @property
    def seed(self):
        return self._seed

    @property
    def scale_factor(self):
        return self._scale_factor

    @property
    def tm(self):
        return self._tm

    @tm.setter
    def tm(self, other_tm):
        self._tm = other_tm
        np.fill_diagonal(self._tm, 0.0)
        self._problem._invalidate_commodity_lists()

    @property
    def total_demand(self):
        return np.sum(self._tm)

    @property
    def fullness(self):
        return np.count_nonzero(self._tm) / self._tm.size

    @property
    def is_full(self):
        return np.alltrue(
            np.argwhere(self._tm == 0.0)
            == np.array([[i, i] for i in range(self.tm.shape[0])])
        )

    def serialize(self, dir_path, fmt="pickle"):
        if fmt == "pickle":
            with open(
                os.path.join(dir_path, "{}_traffic-matrix.pkl".format(self._fname)),
                "wb",
            ) as w:
                pickle.dump(self.tm, w)
        elif fmt == "text":
            np.savetxt(
                "{}_traffic-matrix.txt".format(self._fname),
                self.tm,
                fmt="%10.7f",
                delimiter=" ",
            )
        else:
            raise Exception('"{}" not a valid serialization format'.format(fmt))

    @classmethod
    def from_file(cls, fname):
        if fname.endswith(".pkl"):
            with open(fname, "rb") as f:
                tm = pickle.load(f)
        elif fname.endswith(".txt"):
            tm = np.loadtxt(fname)
        else:
            raise Exception('"{}" not a valid file format'.format(fname))

        vals = os.path.basename(fname)[:-4].split("_")
        model, seed, scale_factor = vals[1], int(vals[2]), float(vals[3])
        vals = vals[4:]

        if model == "gravity":
            random = True if vals[0] == "True" else False
            return GravityTrafficMatrix(
                problem=None,
                tm=tm,
                total_demand=None,
                random=random,
                seed=seed,
                scale_factor=scale_factor,
            )
        elif model == "uniform":
            return UniformTrafficMatrix(
                problem=None,
                tm=tm,
                max_demand=None,
                seed=seed,
                scale_factor=scale_factor,
            )
        elif model == "real":
            date, time = vals[0], vals[1]
            return RealTrafficMatrix(
                problem=None,
                tm=tm,
                date=date,
                time=time,
                seed=seed,
                scale_factor=scale_factor,
            )
        elif model == "bimodal":
            fraction = float(vals[0])
            low_range = tuple([float(x) for x in vals[1].split("-")])
            high_range = tuple([float(x) for x in vals[2].split("-")])
            return BimodalTrafficMatrix(
                problem=None,
                tm=tm,
                fraction=fraction,
                low_range=low_range,
                high_range=high_range,
                seed=seed,
                scale_factor=scale_factor,
            )
        elif model == "behnaz":
            return BehnazTrafficMatrix(
                problem=None, tm=tm, seed=seed, scale_factor=scale_factor
            )
        elif model == "poisson":
            lam, decay, const_factor = float(vals[0]), float(vals[1]), float(vals[2])
            return PoissonTrafficMatrix(
                problem=None,
                tm=tm,
                lam=lam,
                decay=decay,
                const_factor=const_factor,
                seed=seed,
                scale_factor=scale_factor,
            )
        elif model == "exponential":
            decay, const_factor = float(vals[0]), float(vals[1])
            return ExponentialTrafficMatrix(
                problem=None,
                tm=tm,
                decay=decay,
                const_factor=const_factor,
                seed=seed,
                scale_factor=scale_factor,
            )

    def perturb_matrix(self, mean, stddev):
        self._tm += np.random.normal(mean, stddev, self._tm.shape)
        np.fill_diagonal(self._tm, 0.0)
        self._tm[self._tm < 0.0] = 0.0  # demands can never be less than 0

    def perturb_matrix_mult(self, mean, stdev, seed_prob_tm):
        self._tm *= 1 + np.random.choice([-1, 1]) * np.random.normal(mean, stdev)
        np.fill_diagonal(
            self._tm, 0.0
        )  # should not be needed if self._tm has a zero diagonal
        too_low_indexes = self._tm < 0.1 * seed_prob_tm
        self._tm[too_low_indexes] = seed_prob_tm[too_low_indexes]

    def update_matrix(self, scale_factor, type, **kwargs):
        self._seed += 1
        self._scale_factor = scale_factor
        self._update(type, **kwargs)
        self._tm *= self.scale_factor
        self.problem._invalidate_commodity_lists()

    @property
    def _fname(self):
        return "{}_{}_{}_{}_{}".format(
            self.problem.name,
            self.model,
            self.seed,
            self.scale_factor,
            self._fname_suffix,
        )

    @property
    def model(self):
        raise NotImplementedError(
            "@property model needs to be implemented in the subclass: {}".format(
                self.__class__
            )
        )

    def copy(self):
        raise NotImplementedError(
            "copy needs to be implemented in the subclass: {}".format(self.__class__)
        )

    def _init_traffic_matrix(self):
        raise NotImplementedError(
            "_init_traffic_matrix needs to be implemented in the subclass: {}".format(
                self.__class__
            )
        )

    def _update(self, type, **kwargs):
        raise NotImplementedError(
            "update_matrix needs to be implemented in the subclass: {}".format(
                self.__class__
            )
        )

    @property
    def _fname_suffix(self):
        raise NotImplementedError(
            "@property _fname_suffix needs to be implemented in the subclass: {}".format(
                self.__class__
            )
        )


class GenericTrafficMatrix(TrafficMatrix):
    def __init__(self, problem, tm, scale_factor=1.0):
        assert tm is not None
        super().__init__(problem, tm, seed=0, scale_factor=scale_factor)

    @property
    def model(self):
        return "generic"

    def copy(self):
        return GenericTrafficMatrix(self.problem, self._tm.copy())

    def _init_traffic_matrix(self):
        pass

    def _update(self, type, **kwargs):
        pass

    @property
    def _fname_suffix(self):
        return ""


class GravityTrafficMatrix(TrafficMatrix):
    def __init__(
        self, problem, tm, total_demand, random=False, seed=0, scale_factor=1.0
    ):
        if tm is not None:
            self._total_demand = np.sum(tm)
        else:
            self._total_demand = total_demand
        self._random = random
        super().__init__(problem, tm, seed, scale_factor)

    @property
    def model(self):
        return "gravity"

    @property
    def random(self):
        return self._random

    def copy(self):
        return GravityTrafficMatrix(
            self.problem,
            self._tm.copy(),
            self.total_demand,
            self.random,
            self.seed,
            self.scale_factor,
        )

    def _init_traffic_matrix(self):
        np.random.seed(self.seed)
        num_nodes = len(self.problem.G.nodes)
        self._tm = np.zeros((num_nodes, num_nodes), dtype=np.float32)

        sccs = nx.strongly_connected_components(self.problem.G)
        for scc in sccs:
            in_cap_sum, out_cap_sum = defaultdict(float), defaultdict(float)
            for u in scc:
                for v in self.problem.G.predecessors(u):
                    in_cap_sum[u] += self.problem.G[v][u]["capacity"]
                for v in self.problem.G.successors(u):
                    out_cap_sum[u] += self.problem.G[u][v]["capacity"]
            in_cap_sum, out_cap_sum = dict(in_cap_sum), dict(out_cap_sum)

            in_total_cap = sum(in_cap_sum.values())
            out_total_cap = sum(out_cap_sum.values())

            for u in scc:
                norm_u = out_cap_sum[u] / out_total_cap
                for v in scc:
                    if u == v:
                        continue
                    frac = norm_u * in_cap_sum[v] / (in_total_cap - in_cap_sum[u])
                    if self.random:
                        # sample from gaussian with mean = frac, stddev = frac / 4
                        self._tm[u, v] = max(np.random.normal(frac, frac / 4), 0.0)
                    else:
                        self._tm[u, v] = frac

        self._tm *= self._total_demand

    def _update(self, _=None):
        self._init_traffic_matrix()

    @property
    def _fname_suffix(self):
        return "{}_{}".format(self.total_demand, self.random)


class UniformTrafficMatrix(TrafficMatrix):
    def __init__(self, problem, tm, max_demand, seed=0, scale_factor=1.0):
        if tm is not None:
            self._max_demand = np.max(tm)
        else:
            self._max_demand = max_demand
        super().__init__(problem, tm, seed, scale_factor)

    @property
    def max_demand(self):
        return self._max_demand

    @property
    def model(self):
        return "uniform"

    def copy(self):
        return UniformTrafficMatrix(
            self.problem, self._tm.copy(), self.max_demand, self.seed, self.scale_factor
        )

    def _init_traffic_matrix(self):
        np.random.seed(self.seed)
        num_nodes = len(self.problem.G.nodes)

        self._tm = np.random.rand(num_nodes, num_nodes) * self._max_demand
        self._tm = self._tm.astype(np.float32)
        np.fill_diagonal(self._tm, 0.0)

    def _update(self, _=None):
        self._init_traffic_matrix()

    @property
    def _fname_suffix(self):
        return str(self.max_demand)


class ExponentialTrafficMatrix(TrafficMatrix):
    def __init__(
        self, problem, tm, beta, decay, const_factor, seed=0, scale_factor=1.0
    ):
        assert decay <= 1.0
        self._beta = beta
        self._decay = decay
        self._const_factor = const_factor
        super().__init__(problem, tm, seed, scale_factor)

    @property
    def beta(self):
        return self._beta

    @property
    def decay(self):
        return self._decay

    @property
    def const_factor(self):
        return self._const_factor

    @property
    def model(self):
        return "exponential"

    def copy(self):
        return ExponentialTrafficMatrix(
            self.problem,
            self._tm.copy(),
            self.beta,
            self.decay,
            self.const_factor,
            self.seed,
            self.scale_factor,
        )

    def _init_traffic_matrix(self):
        G = self.problem.G
        np.random.seed(self.seed)
        num_nodes = len(G.nodes)

        distances = np.zeros((num_nodes, num_nodes), dtype=np.int)
        dist_iter = nx.shortest_path_length(G)
        for src, dist_dict in dist_iter:
            for target, dist in dist_dict.items():
                distances[src, target] = dist

        self._tm = np.array(
            [
                [
                    np.random.exponential(self._beta * (self._decay ** dist))
                    for dist in row
                ]
                for row in distances
            ],
            dtype=np.float32,
        )
        # No traffic between node and itself
        np.fill_diagonal(self._tm, 0.0)

        self._tm *= self._const_factor

    def _update(self, scale_factor, _=None):
        self._init_traffic_matrix(scale_factor)

    @property
    def _fname_suffix(self):
        return "{}_{}_{}".format(self.beta, self.decay, self.const_factor)


class PoissonTrafficMatrix(TrafficMatrix):
    def __init__(self, problem, tm, lam, decay, const_factor, seed=0, scale_factor=1.0):
        assert decay <= 1.0
        self._lam = lam
        self._decay = decay
        self._const_factor = const_factor
        super().__init__(problem, tm, seed, scale_factor)

    @property
    def lam(self):
        return self._lam

    @property
    def decay(self):
        return self._decay

    @property
    def const_factor(self):
        return self._const_factor

    @property
    def model(self):
        return "poisson"

    def copy(self):
        return PoissonTrafficMatrix(
            self.problem,
            self._tm.copy(),
            self.lam,
            self._decay,
            self._const_factor,
            self.seed,
            self.scale_factor,
        )

    def _init_traffic_matrix(self):
        G = self.problem.G
        np.random.seed(self.seed)
        num_nodes = len(G.nodes)

        distances = np.zeros((num_nodes, num_nodes), dtype=np.int)
        dist_iter = nx.shortest_path_length(G)
        for src, dist_dict in dist_iter:
            for target, dist in dist_dict.items():
                distances[src, target] = dist

        self._tm = np.array(
            [
                [np.random.poisson(self._lam * (self._decay ** dist)) for dist in row]
                for row in distances
            ],
            dtype=np.float32,
        )
        # No traffic between node and itself
        np.fill_diagonal(self._tm, 0.0)

        self._tm *= self._const_factor

    def _update(self, _=None):
        self._init_traffic_matrix()

    @property
    def _fname_suffix(self):
        return "{}_{}_{}".format(self.lam, self.decay, self.const_factor)


class GaussianTrafficMatrix(TrafficMatrix):
    def __init__(self, problem, tm, mean, stddev, seed=0, scale_factor=1.0):
        self._mean = mean
        self._stddev = stddev
        super().__init__(problem, tm, seed, scale_factor)

    @property
    def mean(self):
        return self._mean

    @property
    def stddev(self):
        return self._stddev

    @property
    def model(self):
        return "gaussian"

    def copy(self):
        return GaussianTrafficMatrix(
            self.problem,
            self._tm.copy(),
            self.mean,
            self.stddev,
            self.seed,
            self.scale_factor,
        )

    def _init_traffic_matrix(self):
        G = self.problem.G
        np.random.seed(self.seed)
        num_nodes = len(G.nodes)

        self._tm = np.random.normal(self.mean, self.stddev, (num_nodes, num_nodes))
        self._tm[self._tm < 0.0] = 0.0
        # No traffic between node and itself
        np.fill_diagonal(self._tm, 0.0)

    def _update(self, _=None):
        self._init_traffic_matrix()

    @property
    def _fname_suffix(self):
        return "{}_{}".format(self.mean, self.stddev)


# Bimodal distribution, made up of two uniform distributions. `fraction`
# entries sampled uniformly from [high_range[0], high_range[1]], and 1 -
# `fraction` entries sampled uniformly from [low_range[0], low_range[1]]
class BimodalTrafficMatrix(TrafficMatrix):
    def __init__(
        self, problem, tm, fraction, low_range, high_range, seed=0, scale_factor=1.0
    ):
        assert 0.0 <= low_range[0] < low_range[1] < high_range[0] < high_range[1]
        # assert low_range[0] >= 0.0
        # assert low_range[0] < low_range[1]
        # assert low_range[1] < high_range[0]
        # assert high_range[0] < high_range[1]

        self._fraction = fraction
        self._low_range = low_range
        self._high_range = high_range
        super().__init__(problem, tm, seed, scale_factor)

    @property
    def fraction(self):
        return self._fraction

    @property
    def low_range(self):
        return self._low_range

    @property
    def high_range(self):
        return self._high_range

    @property
    def model(self):
        return "bimodal"

    def copy(self):
        return BimodalTrafficMatrix(
            self.problem,
            self._tm.copy(),
            self.fraction,
            self.low_range,
            self.high_range,
            self.seed,
            self.scale_factor,
        )

    def _init_traffic_matrix(self):
        G = self.problem.G
        np.random.seed(self.seed)
        num_nodes = len(G.nodes)

        self._tm = np.zeros((num_nodes, num_nodes))
        inds = np.random.choice(
            2, (num_nodes, num_nodes), p=[self.fraction, 1 - self.fraction]
        ).astype("bool")
        self._tm[inds] = np.random.uniform(
            self.low_range[0], self.low_range[1], np.sum(inds)
        )
        self._tm[~inds] = np.random.uniform(
            self.high_range[0], self.high_range[1], np.sum(~inds)
        )
        # No traffic between node and itself
        np.fill_diagonal(self._tm, 0.0)

    def _update(self, scale_factor, _=None):
        self._init_traffic_matrix(scale_factor)

    @property
    def _fname_suffix(self):
        return "{}_{}-{}_{}-{}".format(
            self.fraction,
            self.low_range[0],
            self.low_range[1],
            self.high_range[0],
            self.high_range[1],
        )


class RealTrafficMatrix(TrafficMatrix):
    def __init__(self, problem, tm, date, time, seed=0, scale_factor=1.0):
        super().__init__(problem, tm, seed, scale_factor=1.0)
        self._date = date
        self._time = time

    @property
    def model(self):
        return "real"

    @property
    def date(self):
        return self._date

    @property
    def time(self):
        return self._time

    def copy(self):
        return RealTrafficMatrix(
            self.problem,
            self._tm.copy(),
            self.date,
            self.time,
            self.seed,
            self.scale_factor,
        )

    def _init_traffic_matrix(self):
        pass

    def _update(self, scale_factor, type, **kwargs):
        if type == "uniform":
            alpha = kwargs["alpha"]
            assert alpha > 0.0

            def new_val(val):
                # w~Uni(-1, 1)
                # perturb each demand by +/- alpha * |w|
                return max(0, val * (1 + alpha * uni_rand(-1, 1)))

        elif type == "scale":
            assert scale_factor > 0.0

            def new_val(val):
                # scale each demand by `scale`
                return val * scale_factor

        else:
            raise Exception(
                '"{}" not a valid perturbation type for the traffic matrix'.format(type)
            )

        mat = np.zeros_like(self.tm)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                mat[i, j] = new_val(self._tm[i, j])
        self._tm = mat

    @property
    def _fname_suffix(self):
        return "{}_{}".format(self.date, self.time)
