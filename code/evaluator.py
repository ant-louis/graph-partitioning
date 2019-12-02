class Evaluator:
    def __init__(self, solver, gridParams):
        self.solver = solver
        self.gridParams = gridParams

    def gridSearch(self, dumpOutputBest=True):

        # TODO: iterate over gridParam
        output = self.solver.algo1()
        print("Best parameters were {} with algo {}")
        self.solver.dumpOutput("{}-{}-{}".format(self.solver.graph.name, 1, 2),
                               output)

