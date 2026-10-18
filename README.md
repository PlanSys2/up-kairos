# up-kairos

[![CI](https://github.com/PlanSys2/up-kairos/actions/workflows/ci.yml/badge.svg)](https://github.com/PlanSys2/up-kairos/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/up-kairos?logo=pypi&logoColor=white&label=PyPI)](https://pypi.org/project/up-kairos/)
[![Python versions](https://img.shields.io/pypi/pyversions/up-kairos?logo=python&logoColor=white)](https://pypi.org/project/up-kairos/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

[Kairos](https://github.com/PlanSys2/kairos), a temporal and numeric PDDL planner, as an engine
of the [Unified Planning Framework](https://github.com/aiplan4eu/unified-planning) (UPF).

Kairos plans in the same process (no files, no subprocess), anytime, and proves when no plan
exists. Every plan is validated before it is returned.

```bash
pip install up-kairos
```

```python
import up_kairos
from unified_planning.shortcuts import OneshotPlanner

up_kairos.register()  # until UPF lists Kairos among its engines ("kairos", "kairos-val")

with OneshotPlanner(name="kairos") as planner:
    result = planner.solve(problem, timeout=10)
    print(result.status, result.plan)
```

## Operation modes

| UPF | Kairos |
|---|---|
| `OneshotPlanner` | a plan; with a quality metric and a timeout, the best within it |
| `AnytimePlanner` | each better plan as it comes (increasing quality), then the best |
| `PlanValidator` | `kairos-val`, Kairos' validator: numeric, temporal, PDDL3 (not PDDL+) |
| `Replanner` | through UPF's meta engine: `Replanner(problem, name="replanner[kairos]")` |

Results:

- `SOLVED_OPTIMALLY` when Kairos proves the plan optimal for the problem's metric.
- `UNSOLVABLE_PROVEN` when it proves there is no plan: the search space runs out, the timed
  literals take what the goal needs too soon, or there is no plan even without durations.
- `UNSOLVABLE_INCOMPLETELY` when the engines that apply ran out of states without being
  complete for the task.
- `TIMEOUT`, `MEMOUT`, `UNSUPPORTED_PROBLEM`.

Kairos' diagnostics, including why there is no plan, are in `result.log_messages`. Its
statistics are in `result.metrics`: expanded nodes, states, plans and peak memory.

## Problems

Kairos plans with:

- **Classical planning**, with negative, disjunctive, existential and universal conditions and
  conditional effects.
- **Numeric planning**, linear or not, with bounded numeric fluents and undefined values.
- **Object fluents.**
- **Temporal planning**, with required concurrency, timed effects (timed initial literals and
  fluents), duration inequalities and durations that read fluents.
- **Continuous effects, processes and events** (PDDL+).
- **Trajectory constraints and state invariants.**
- **Quality metrics**: action costs, plan length, makespan, final value, and oversubscription.

It plans 52 of the 66 example problems of UPF 1.3. The rest are hierarchical, scheduling,
integer or boolean parameters, intermediate conditions, timed goals and undefined booleans.
It passes UPF's engine tests (`up_test_cases/report.py kairos kairos-val`: oneshot, anytime
and validation).

Notes:

- **Writing for Kairos.** Some things go beyond what UPF's PDDL writer writes, so this engine
  writes them itself:
  - object fluents, as PDDL 3.1;
  - oversubscription, as goal preferences and a metric of the gain lost;
  - bounded fluents, as state invariants.
- **One instance at a time.** A durative action does not overlap itself, as in POPF:
  `SELF_OVERLAPPING` is not declared.
- **Simultaneous happenings.** Kairos plans with epsilon between happenings that interfere
  (`problem.epsilon`, 0.001 by default), as PDDL 2.1 and VAL. `kairos-val` validates with the
  problem's epsilon, or, without one, as UPF: a happening may read what another at the same
  instant writes, but writes that disagree are invalid.
- **PDDL+ plans and their end.** A PDDL+ plan ends when the goal is checked, after letting
  processes and events act. UPF's plans have no end time, so it comes in
  `result.metrics["kairos_plan_end"]`, and `kairos-val` does not take PDDL+ problems.

## Options

`OneshotPlanner(name="kairos", params={...})`:

- `anytime`
- `timeout`: the default when the call gives none.
- `threads`
- `memory_limit_mb`
- `engine`: `"auto"`, `"m1"`, `"m2"` or `"m3"`.
- `search`
- `seed`
- `stop_after_no_improvement`
- `target_cost`

## License

Apache-2.0.
