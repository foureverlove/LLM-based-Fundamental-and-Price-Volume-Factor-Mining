from factor.factor import Factor

class MiningEngine:

    def __init__(self, generator, executor, evaluator, selector):
        self.generator = generator
        self.executor = executor
        self.evaluator = evaluator
        self.selector = selector
        self.pool = []

    def step(self, data, returns):
        expr = self.generator()
        factor = Factor(expr)

        values = self.executor.run(expr, data)
        factor.ic_mean, factor.ic_ts = self.evaluator.calc_ic(values, returns)
        factor.pnl = self.evaluator.simple_long_short(values, returns)

        score = self.selector.score(factor, self.pool)
        self.pool.append(factor)

        return factor, score
