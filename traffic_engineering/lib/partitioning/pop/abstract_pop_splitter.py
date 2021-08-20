class AbstractPOPSplitter(object):
    def __init__(self, num_subproblems):
        self._num_subproblems = num_subproblems

    #################
    # Public method #
    #################
    @property
    def split(self):
        raise NotImplementedError(
            "split needs to be implemented in the subclass: {}".format(self.__class__)
        )
