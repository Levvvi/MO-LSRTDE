from __future__ import annotations

from molsrtde.algorithm import MOLSRTDE
from molsrtde.problems import ZDT1


def main() -> None:
    problem = ZDT1(n_var=5)
    result = MOLSRTDE(pop_size=10, max_evals=30, seed=0).run(problem)
    print(result.F.shape)


if __name__ == "__main__":
    main()

