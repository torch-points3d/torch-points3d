from time import time
from collections import defaultdict
import functools
from .running_stats import RunningStats

FunctionStats: defaultdict = defaultdict(RunningStats)


def time_func(*outer_args, **outer_kwargs):
    print_rec = outer_kwargs.get("print_rec", 100)
    measure_runtime = outer_kwargs.get("measure_runtime", False)
    name = outer_kwargs.get("name", "")

    def time_func_inner(func):
        @functools.wraps(func)
        def func_wrapper(*args, **kwargs):
            if measure_runtime:
                func_name = name if name else func.__name__
                if FunctionStats.get(func_name, None) is not None:
                    if FunctionStats[func_name].n % print_rec == 0:
                        stats = FunctionStats[func_name]
                        stats_mean = stats.mean()
                        print(
                            "{} run in {} | {} over {} runs".format(
                                func_name, stats_mean, stats_mean * stats.n, stats.n
                            )
                        )
                        # print('{} run in {} +/- {} over {} runs'.format(func.__name__, stats.mean(), stats.std(), stats.n))
                t0 = time()
                out = func(*args, **kwargs)
                diff = time() - t0
                FunctionStats[func_name].push(diff)
                return out
            else:
                return func(*args, **kwargs)

        return func_wrapper

    return time_func_inner


@time_func(print_rec=50, measure_runtime=True)
def do_nothing():
    pass


def iteration():
    for _ in range(10000):
        do_nothing()


if __name__ == "__main__":
    iteration()
