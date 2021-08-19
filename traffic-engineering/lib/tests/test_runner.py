#! /usr/bin/env python


from .toy_problem_test import ToyProblemTest
from .reconciliation_problem_test import ReconciliationProblemTest
from .reconciliation_problem_2_test import ReconciliationProblem2Test
from .recon3_test import Recon3Test
from .optgapc1_test import OptGapC1Test
from .optgapc2_test import OptGapC2Test
from .optgapc3_test import OptGapC3Test
from .optgap4_test import OptGap4Test
from .single_edge_b import SingleEdgeBTest
from .feasibility_test import FeasibilityTest
from .flow_path_construction_test import FlowPathConstructionTest
from .we_need_to_fix_this_test import WeNeedToFixThisTest
from .abstract_test import bcolors


import argparse


ALL_TESTS = [
    ToyProblemTest(),
    ReconciliationProblemTest(),
    ReconciliationProblem2Test(),
    Recon3Test(),
    OptGapC1Test(),
    OptGapC2Test(),
    OptGapC3Test(),
    FeasibilityTest(),
    OptGap4Test(),
    FlowPathConstructionTest(),
    WeNeedToFixThisTest(),
    SingleEdgeBTest(),
]
TEST_NAME_DICT = {test.name: test for test in ALL_TESTS}


def run_tests(tests_to_run):
    tests_that_failed = []
    for test in tests_to_run:
        print("\n\n---{} TEST---\n\n".format(test.name.upper()))
        test.run()
        if test.has_error:
            tests_that_failed.append(test)
    for test in tests_that_failed:
        print()
        print(
            bcolors.ERROR
            + "\n\n---{} TEST failed---\n\n".format(test.name.upper())
            + bcolors.ENDC
        )
    if len(tests_that_failed) == 0:
        print(bcolors.OKGREEN + "All tests passed!" + bcolors.ENDC)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests", nargs="+", required=False)
    args = parser.parse_args()

    if args.tests is not None:
        tests_to_run = [TEST_NAME_DICT[name] for name in args.tests]
    else:
        tests_to_run = ALL_TESTS

    print(
        "RUNNING THE FOLLOWING TESTS: {}".format([test.name for test in tests_to_run])
    )
    run_tests(tests_to_run)
