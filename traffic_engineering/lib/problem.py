import re
import os
import json
import numpy as np
import networkx as nx
from networkx.readwrite import json_graph
from numbers import Real
from .graph_utils import commodity_gen
from .traffic_matrix import *


class Problem(object):
    def __init__(
        self,
        G,
        traffic_matrix=None,
        model="gravity",
        seed=0,
        scale_factor=1.0,
        **kwargs
    ):
        self.G = G
        self.capacity_seed = seed

        if traffic_matrix is not None:
            if isinstance(traffic_matrix, np.ndarray):
                assert traffic_matrix.shape[0] == len(G)
                self.traffic_matrix = GenericTrafficMatrix(
                    self, traffic_matrix, scale_factor=scale_factor
                )
            elif isinstance(traffic_matrix, TrafficMatrix):
                assert traffic_matrix.tm.shape[0] == len(G)
                self.traffic_matrix = traffic_matrix
                self.traffic_matrix._problem = self
            else:
                raise Exception(
                    '"{}" is an invalid type for traffic matrix'.format(
                        traffic_matrix.__class__
                    )
                )
        elif model == "gravity":
            try:
                self.traffic_matrix = GravityTrafficMatrix(
                    self,
                    tm=None,
                    total_demand=kwargs["total_demand"],
                    random=kwargs["random"],
                    seed=seed,
                    scale_factor=scale_factor,
                )
            except KeyError as e:
                raise e
        elif model == "uniform":
            try:
                self.traffic_matrix = UniformTrafficMatrix(
                    self,
                    tm=None,
                    max_demand=kwargs["max_demand"],
                    seed=seed,
                    scale_factor=scale_factor,
                )
            except KeyError as e:
                raise e
        elif model == "poisson":
            try:
                self.traffic_matrix = PoissonTrafficMatrix(
                    self,
                    tm=None,
                    decay=kwargs["decay"],
                    lam=kwargs["lam"],
                    const_factor=kwargs["const_factor"],
                    seed=seed,
                    scale_factor=scale_factor,
                )
            except KeyError as e:
                raise e
        elif model == "gaussian":
            try:
                self.traffic_matrix = GaussianTrafficMatrix(
                    self,
                    tm=None,
                    mean=kwargs["mean"],
                    stddev=kwargs["stddev"],
                    seed=seed,
                    scale_factor=scale_factor,
                )
            except KeyError as e:
                raise e
        elif model == "bimodal":
            try:
                self.traffic_matrix = BimodalTrafficMatrix(
                    self,
                    tm=None,
                    fraction=kwargs["fraction"],
                    low_range=kwargs["low_range"],
                    high_range=kwargs["high_range"],
                    seed=seed,
                    scale_factor=scale_factor,
                )
            except KeyError as e:
                raise e
        elif model == "exponential":
            try:
                self.traffic_matrix = ExponentialTrafficMatrix(
                    self,
                    tm=None,
                    decay=kwargs["decay"],
                    const_factor=kwargs["const_factor"],
                    seed=seed,
                    scale_factor=scale_factor,
                )
            except KeyError as e:
                raise e
        elif model == "behnaz":
            self.traffic_matrix = BehnazTrafficMatrix(
                self, tm=None, scale_factor=scale_factor, **kwargs
            )
        else:
            raise Exception(
                'no traffic matrix passed in; model argument "{}" is invalid'.format(
                    model
                )
            )

    ############################
    # Public class methods:    #
    # Instantiate new Problems #
    ############################
    @classmethod
    def from_file(cls, topology_fname, traffic_matrix_fname, old_way=True):
        if topology_fname.endswith("-edgelist.txt"):
            G = nx.read_edgelist(
                topology_fname,
                create_using=nx.DiGraph,
                nodetype=int,
                data=[("capacity", float)],
            )
        elif topology_fname.endswith(".json"):
            G = Problem._read_graph_json(topology_fname)
            old_G = None
        elif topology_fname.endswith(".graphml"):
            G = Problem._read_graph_graphml(topology_fname)
            old_G = None
        elif topology_fname.endswith(".dot"):
            G = Problem._read_graph_dot(topology_fname, old_way=old_way)
            old_G = Problem._read_graph_dot(topology_fname, old_way=True)

        tm = TrafficMatrix.from_file(traffic_matrix_fname)
        problem = cls(G=G, traffic_matrix=tm, seed=0)
        if old_G is not None:
            problem.old_G = old_G
        problem.name = os.path.basename(topology_fname)

        return problem

    @classmethod
    def fixed_traffic_matrix_problem(cls, G, traffic_matrix, seed=0):
        problem = cls(G=G, traffic_matrix=traffic_matrix, seed=seed)
        return problem

    ###########################
    # Public instance methods #
    ###########################
    def print_stats(self):
        print("Num nodes: ", len(self.G.nodes))
        print("Num edges: ", len(self.G.edges))
        print("Num commodities: ", len(self.commodity_list))

    def copy(self):
        G = self.G.copy()
        tm = self.traffic_matrix.copy()
        problem = Problem(G, tm)
        problem.name = self.name
        tm._problem = problem
        problem.capacity_seed = self.capacity_seed
        return problem

    ##############
    # Properties #
    ##############
    @property
    def G(self):
        return self._G

    @G.setter
    def G(self, G):
        self._G = G

    @property
    def traffic_matrix(self):
        return self._traffic_matrix

    def _invalidate_commodity_lists(self):
        if hasattr(self, "_commodity_list"):
            del self._commodity_list
        if hasattr(self, "_multi_commodity_list"):
            del self._multi_commodity_list
        if hasattr(self, "_sparse_commodity_list"):
            del self._sparse_commodity_list

    @traffic_matrix.setter
    def traffic_matrix(self, traffic_matrix):
        self._traffic_matrix = traffic_matrix
        # invalidate commodity list attributes, since we've updated the traffic
        # matrix
        self._invalidate_commodity_lists()

    @property
    def capacity_seed(self):
        return self._capacity_seed

    @capacity_seed.setter
    def capacity_seed(self, capacity_seed):
        self._capacity_seed = capacity_seed

    # set both traffic seed and capacity seed to the same seed; otherwise
    # invoke the normal setter function
    def __setattr__(self, name, value):
        if name == "seed":
            super().__setattr__("_capacity_seed", value)
        else:
            super().__setattr__(name, value)

    @property
    def edges_list(self):
        if not hasattr(self, "_edges_list"):
            self._edges_list = list(self.G.edges)
        return self._edges_list

    @property
    def commodity_list(self):
        if not hasattr(self, "_commodity_list"):
            self._commodity_list = list(
                enumerate(commodity_gen(self.traffic_matrix.tm))
            )
        return self._commodity_list

    @property
    def multi_commodity_list(self):
        if not hasattr(self, "_multi_commodity_list"):
            self._multi_commodity_list = [
                (k, [x], [y], z)
                for k, (x, y, z) in enumerate(commodity_gen(self.traffic_matrix.tm))
            ]
        return self._multi_commodity_list

    @property
    def sparse_commodity_list(self):
        if not hasattr(self, "_sparse_commodity_list"):
            self._sparse_commodity_list = list(
                enumerate(commodity_gen(self.traffic_matrix.tm, skip_zero=False))
            )
        return self._sparse_commodity_list

    @property
    def edge_idx(self):
        return {edge: e for e, edge in enumerate(self.G.edges)}

    def new_capacities(self, *, min_cap, max_cap, fixed_caps=[], same_both_ways=True):
        assert isinstance(min_cap, Real)
        assert isinstance(max_cap, Real)
        assert max_cap >= min_cap
        self.capacity_seed += 1
        self._change_capacities(
            min_cap=min_cap,
            max_cap=max_cap,
            fixed_caps=fixed_caps,
            same_both_ways=same_both_ways,
        )

    def intra_and_inter_demands(self, partitioner):
        p_v = partitioner.partition(self)
        intra, inter = 0.0, 0.0
        for _, (s_k, t_k, d_k) in self.commodity_list:
            if p_v[s_k] == p_v[t_k]:
                intra += d_k
            else:
                inter += d_k
        return intra, inter

    @property
    def is_traffic_matrix_full(self):
        return (
            len(self.commodity_list)
            == self.traffic_matrix.tm.size - self.traffic_matrix.tm.shape[0]
        )

    @property
    def total_demand(self):
        return self.traffic_matrix.total_demand

    @property
    def total_capacity(self):
        return sum(cap for _, _, cap in self.G.edges.data("capacity"))

        ##########################

    # Private helper methods #
    ##########################

    def _change_capacities(
        self, *, min_cap, max_cap, fixed_caps=[], same_both_ways=True
    ):
        assert isinstance(min_cap, Real)
        assert isinstance(max_cap, Real)
        assert max_cap >= min_cap
        # reset the seed before we generate the edge capacities
        np.random.seed(self.capacity_seed)

        # Sample evenly spaced floating-point numbers between a and b
        # Borrowed from https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.random.random_integers.html
        def sample_from_range(a, b):
            N = int(b - a) + 1
            if N == 1:
                return a + (b - a) * np.random.random_integers(N)
            return a + (b - a) * (np.random.random_integers(N) - 1) / (N - 1.0)

        for u, v in self.G.edges():
            if "capacity" not in self.G[u][v]:
                cap = sample_from_range(min_cap, max_cap)
                self.G[u][v]["capacity"] = float(cap)
                if same_both_ways and "capacity" not in self.G[v][u]:
                    self.G[v][u]["capacity"] = float(cap)

        for (u, v, cap) in fixed_caps:
            self.G[u][v]["capacity"] = float(cap)
            if same_both_ways:
                self.G[v][u]["capacity"] = float(cap)

    ##########################
    # Private static methods #
    ##########################
    @staticmethod
    def _read_graph_json(fname):
        assert fname.endswith(".json")
        with open(fname) as f:
            data = json.load(f)
        return json_graph.node_link_graph(data)

    @staticmethod
    def _write_graph_json(G, fname):
        assert fname.endswith(".json")
        with open(fname, "w") as w:
            json.dump(json_graph.node_link_data(G), w)

    @staticmethod
    def _read_graph_graphml(fname):
        assert fname.endswith(".graphml")
        file_G = nx.read_graphml(fname).to_directed()
        if isinstance(file_G, nx.MultiDiGraph):
            file_G = nx.DiGraph(file_G)

        G = []
        # Pick largest strongly connected component
        for scc_ids in nx.strongly_connected_components(file_G):
            scc = file_G.subgraph(scc_ids)
            if len(scc) > len(G):
                G = scc
        G = nx.convert_node_labels_to_integers(G)
        # For TZ topologies, assume every link has 1000 Mbps of capacity
        for u, v in G.edges():
            G[u][v]["capacity"] = 1000.0
        return G

    @staticmethod
    # Read Topology Zoo dot file
    def _read_graph_dot(fname, old_way=True):
        assert fname.endswith(".dot")
        if old_way:
            G = nx.drawing.nx_agraph.read_dot(fname)
            for node in G.nodes:
                G.nodes[node]["name"] = node
            G = nx.convert_node_labels_to_integers(G)
        else:
            G = nx.drawing.nx_agraph.read_dot(fname)
            nodes_to_remove = []
            node_data = {}
            # remove hosts, but copy over their node data
            for node in G.nodes:
                if node.startswith("h"):
                    switch_name = node.replace("h", "s")
                    node_data[switch_name] = G.node[node]
                    node_data[switch_name]["name"] = node
                    nodes_to_remove.append(node)
            G.remove_nodes_from(nodes_to_remove)

            # update switches to be hosts; copy over the host info
            nx.set_node_attributes(G, node_data)
            G = nx.convert_node_labels_to_integers(G)

        for u, v, cap in G.edges.data("capacity"):
            # convert capacity links to be in Mbps
            G[u][v]["capacity"] = float(re.match(r"(\d)Gbps", cap)[1]) * 1e3
        return G

    ###########################
    # Abstract method (sorta) #
    ###########################
    @property
    def name(self):
        if hasattr(self, "_name"):
            return self._name
        raise NotImplementedError(
            "name needs to be implemented in the subclass: {}".format(self.__class__)
        )

    @name.setter
    def name(self, name):
        self._name = name
