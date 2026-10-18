# Copyright 2026 Intelligent Robotics Lab
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""UPF's example problems: every one whose kind Kairos declares is planned, its plan checked
by UPF's own validators, and the example's valid and invalid plans by Kairos' validator."""

import pytest
import unified_planning as up
from unified_planning.engines import PlanGenerationResultStatus as S
from unified_planning.engines import ValidationResultStatus as V
from unified_planning.shortcuts import AnytimePlanner, OneshotPlanner, PlanValidator
from unified_planning.test.examples import get_example_problems

import up_kairos
from up_kairos import KairosEngine

up_kairos.register()
up.shortcuts.get_environment().credits_stream = None

EXAMPLES = get_example_problems()
SUPPORTED = sorted(n for n, t in EXAMPLES.items() if KairosEngine.supports(t.problem.kind))
UNSUPPORTED = sorted(set(EXAMPLES) - set(SUPPORTED))


# Example plans whose verdict differs, and why
SEMANTICS = {
    # Its "valid" plan increases value(o1) from 1 to 2, and the goal is value(o1) == 1 (no UPF
    # validator handles the problem's undefined fluents, so the example is not checked there)
    "basic_undef_numeric": "the example's valid plan does not reach the goal",
    # Happenings at the same instant that interfere (PDDL 2.1, VAL): UPF allows them with no
    # epsilon, Kairos plans and validates with epsilon separation
    "logistic": "load and move of the same robot at the same instant interfere",
    "temporal_conditional": "two ends at the same instant interfere",
}


# Fixed in UPF after 1.3 (the example's plan)
FIXED_LATER = {"basic_undef_numeric"}


def pddl_plus(problem):
    return problem.kind.has_processes() or problem.kind.has_events()


def upf_validator(problem, plan):
    """UPF's validator for the kind of plan (not Kairos')."""
    name = "sequential_plan_validator" if isinstance(plan, up.plans.SequentialPlan) else \
        "up_time_triggered_validator"
    return PlanValidator(name=name)


def test_the_bank_has_what_kairos_declares():
    # Most of UPF's examples; the rest need what PDDL cannot say (or Kairos does not do)
    assert len(SUPPORTED) >= 40, UNSUPPORTED


@pytest.mark.parametrize("name", SUPPORTED)
def test_plans(name):
    case = EXAMPLES[name]
    with OneshotPlanner(name="kairos") as planner:
        result = planner.solve(case.problem, timeout=20)
    if not case.solvable:
        assert result.status in (S.UNSOLVABLE_PROVEN, S.UNSOLVABLE_INCOMPLETELY), result
        return
    assert result.status in (S.SOLVED_SATISFICING, S.SOLVED_OPTIMALLY), \
        f"{result.status}: {result.log_messages}"
    plan = result.plan
    assert plan is not None
    if pddl_plus(case.problem):
        assert "kairos_plan_end" in result.metrics  # UPF's plans have no end of their own
        return
    with upf_validator(case.problem, plan) as v:
        if v.supports(case.problem.kind):
            assert v.validate(case.problem, plan).status == V.VALID, str(plan)
    with PlanValidator(name="kairos") as kv:
        assert kv.validate(case.problem, plan).status == V.VALID
    if case.optimum is not None and result.status == S.SOLVED_OPTIMALLY:
        metric = case.problem.quality_metrics[0]
        with upf_validator(case.problem, plan) as v:
            if v.supports(case.problem.kind):
                value = v.validate(case.problem, plan).metric_evaluations[metric]
                assert value == case.optimum


@pytest.mark.parametrize("name", [
    pytest.param(n, marks=pytest.mark.xfail(reason=SEMANTICS[n], strict=n not in FIXED_LATER))
    if n in SEMANTICS else n for n in SUPPORTED if not pddl_plus(EXAMPLES[n].problem)])
def test_validates_as_upf(name):
    case = EXAMPLES[name]
    with PlanValidator(name="kairos") as kv:
        for plan in case.valid_plans:
            assert kv.validate(case.problem, plan).status == V.VALID, str(plan)
        for plan in case.invalid_plans:
            assert kv.validate(case.problem, plan).status == V.INVALID, str(plan)


@pytest.mark.parametrize("name", [n for n in SUPPORTED if EXAMPLES[n].solvable])
def test_anytime(name):
    case = EXAMPLES[name]
    with AnytimePlanner(name="kairos") as planner:
        results = list(planner.get_solutions(case.problem, timeout=5))
    assert results
    *intermediate, last = results
    assert all(r.status == S.INTERMEDIATE for r in intermediate)
    assert last.status in (S.SOLVED_SATISFICING, S.SOLVED_OPTIMALLY), last
    assert last.plan is not None


def test_pddl_plus_plans_are_not_validated():
    case = EXAMPLES["1d_movement"]
    with OneshotPlanner(name="kairos") as planner:
        plan = planner.solve(case.problem, timeout=10).plan
    with PlanValidator(name="kairos") as kv:
        assert kv.validate(case.problem, plan).status == V.UNKNOWN
