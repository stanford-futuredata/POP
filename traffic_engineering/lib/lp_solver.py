from gurobipy import GurobiError
from enum import Enum, unique
import sys


@unique
class Method(Enum):
    PRIMAL_SIMPLEX = 0
    DUAL_SIMPLEX = 1
    BARRIER = 2
    CONCURRENT = 3
    PRIMAL_AND_DUAL = 4


class LpSolver(object):
    def __init__(
        self, model, debug_fn=None, DEBUG=False, VERBOSE=False, out=None, gurobi_out=""
    ):
        if out is None:
            out = sys.stdout
        self._model = model
        self._debug_fn = debug_fn
        self.DEBUG = DEBUG
        self.VERBOSE = VERBOSE
        self.out = out
        self._gurobi_out = gurobi_out

    def _print(self, *args):
        print(*args, file=self.out)

    @property
    def gurobi_out(self):
        return self._gurobi_out

    @gurobi_out.setter
    def gurobi_out(self, gurobi_out):
        if gurobi_out == "stdout" or gurobi_out == "<stdout>":
            self._gurobi_out = "gurobi.log"
        else:
            self._gurobi_out = gurobi_out

    # Note: this is not idempotent: the `model` parameter will be changed after invoking
    # this function
    def solve_lp(
        self, num_threads=None, bar_tol=None, err_tol=None, numeric_focus=False
    ):
        model = self._model
        if numeric_focus:
            model.setParam("NumericFocus", 1)
        if num_threads:
            model.setParam("Threads", num_threads)
        model.setParam("LogFile", self.gurobi_out)
        try:
            if bar_tol:
                model.Params.BarConvTol = bar_tol
            if err_tol:
                model.Params.OptimalityTol = err_tol
                model.Params.FeasibilityTol = err_tol

            # if self.VERBOSE:
            self._print("\nSolving LP")
            model.optimize()

            if self.DEBUG or self.VERBOSE:
                for var in model.getVars():
                    if var.x != 0:
                        if self.DEBUG and self._debug_fn:
                            if not var.varName.startswith("f["):
                                continue
                            u, v, k, s_k, t_k, d_k = self._debug_fn(var)
                            if self.VERBOSE:
                                self._print(
                                    "edge ({}, {}), demand ({}, ({}, {}, {})), flow: {}".format(
                                        u, v, k, s_k, t_k, d_k, var.x
                                    )
                                )
                        elif self.VERBOSE:
                            self._print("{} {}".format(var.varName, var.x))
                self._print("Obj: %g" % model.objVal)
            return model.objVal
        except GurobiError as e:
            self._print("Error code " + str(e.errno) + ": " + str(e))
        except AttributeError as e:
            self._print(str(e))
            self._print("Encountered an attribute error")

    @property
    def model(self):
        return self._model

    @property
    def obj_val(self):
        return self._model.objVal
