class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    ERROR = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


class AbstractTest(object):
    def __init__(self):
        self.has_error = False

    @property
    def name(self):
        raise NotImplementedError(
            "name needs to be implemented in the subclass: {}".format(self.__class__)
        )

    def assert_feasibility(self, ncflow):
        try:
            ncflow.check_feasibility()
        except AssertionError:
            self.has_error = True
            print(
                bcolors.ERROR
                + "[ERROR] NCFlow did not find feasible flow"
                + bcolors.ENDC
            )

    def assert_eq_epsilon(self, actual_val, correct_val, epsilon=1e-5):
        try:
            assert abs(correct_val - actual_val) < epsilon
        except AssertionError:
            self.has_error = True
            print(
                bcolors.ERROR
                + "[ERROR] Correct value: {}, actual value: {}".format(
                    correct_val, actual_val
                )
                + bcolors.ENDC
            )

    def assert_geq_epsilon(self, actual_val, lower_val, epsilon=1e-5):
        try:
            assert actual_val >= lower_val - epsilon
        except AssertionError:
            self.has_error = True
            print(
                bcolors.ERROR
                + "[ERROR] Looking for >= {}, actual value: {}".format(
                    lower_val, actual_val
                )
                + bcolors.ENDC
            )

    def assert_leq_epsilon(self, actual_val, upper_val, epsilon=1e-5):
        try:
            assert actual_val <= upper_val + epsilon
        except AssertionError:
            self.has_error = True
            print(
                bcolors.ERROR
                + "[ERROR] Looking for <= {}, actual value: {}".format(
                    upper_val, actual_val
                )
                + bcolors.ENDC
            )

    def run(self):
        raise NotImplementedError(
            "run needs to be implemented in the subclass: {}".format(self.__class__)
        )
