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

"""What the Kairos engine adds over a PDDL planner: statuses, proofs, anytime, options, object
fluents, oversubscription, bounded fluents, its validator."""

import io
from fractions import Fraction

import pytest
import unified_planning as up
from unified_planning.engines import PlanGenerationResultStatus as S
from unified_planning.engines import ValidationResultStatus as V
from unified_planning.shortcuts import (
    GE, AnytimePlanner, BoolType, DurativeAction, Fluent, InstantaneousAction, IntType,
    MinimizeActionCosts, MinimizeSequentialPlanLength, Not, Object, OneshotPlanner,
    Oversubscription, PlanValidator, Problem, RealType, Replanner, StartTiming, EndTiming,
    UserType, get_environment)

import up_kairos

up_kairos.register()
get_environment().credits_stream = None


def robots(rooms=3, connect=True):
    """A robot moving along a corridor of rooms; the goal is the last one."""
    Room = UserType("Room")
    at = Fluent("at", BoolType(), r=Room)
    next_to = Fluent("next_to", BoolType(), a=Room, b=Room)
    move = InstantaneousAction("move", a=Room, b=Room)
    a, b = move.parameters
    move.add_precondition(at(a))
    move.add_precondition(next_to(a, b))
    move.add_effect(at(a), False)
    move.add_effect(at(b), True)
    p = Problem("robots")
    p.add_fluent(at, default_initial_value=False)
    p.add_fluent(next_to, default_initial_value=False)
    p.add_action(move)
    rs = [Object(f"r{i}", Room) for i in range(rooms)]
    p.add_objects(rs)
    p.set_initial_value(at(rs[0]), True)
    for x, y in zip(rs, rs[1:]):
        if connect:
            p.set_initial_value(next_to(x, y), True)
    p.add_goal(at(rs[-1]))
    return p, move, rs


def solve(problem, **params):
    with OneshotPlanner(name="kairos", params=params) as planner:
        return planner.solve(problem, timeout=10)


# --- Statuses ------------------------------------------------------------------------------------

def test_solved_and_mapped_back_to_the_problem():
    p, move, rs = robots()
    r = solve(p)
    assert r.status == S.SOLVED_SATISFICING
    assert [ai.action for ai in r.plan.actions] == [move, move]
    assert [ai.actual_parameters[1].object() for ai in r.plan.actions] == rs[1:]
    assert r.engine_name == "kairos"
    assert float(r.metrics["engine_internal_time"]) >= 0


def test_unsolvable_proven():
    p, _, _ = robots(connect=False)
    r = solve(p)
    assert r.status == S.UNSOLVABLE_PROVEN
    assert r.plan is None
    assert any("orchestrator" in m.message for m in r.log_messages)


def test_optimum_proven_with_a_metric():
    p, _, _ = robots(rooms=4)
    p.add_quality_metric(MinimizeSequentialPlanLength())
    r = solve(p)
    assert r.status == S.SOLVED_OPTIMALLY
    assert len(r.plan.actions) == 3


def test_timeout_and_unknown_options():
    p, _, _ = robots()
    with pytest.raises(up.exceptions.UPUsageError):
        solve(p, nonsense=1)
    assert solve(p, threads=2, anytime=False, engine="m1").status == S.SOLVED_SATISFICING


def test_output_stream_receives_the_log():
    p, _, _ = robots()
    out = io.StringIO()
    with OneshotPlanner(name="kairos") as planner:
        planner.solve(p, timeout=10, output_stream=out)
    assert "[orchestrator]" in out.getvalue()


# --- Anytime -------------------------------------------------------------------------------------

def routes():
    """A flight of 10 or six drives of 1: anytime goes from the flight to the drives."""
    Place = UserType("Place")
    at = Fluent("at", BoolType(), p=Place)
    road = Fluent("road", BoolType(), a=Place, b=Place)
    highway = Fluent("highway", BoolType(), a=Place, b=Place)
    drive = InstantaneousAction("drive", a=Place, b=Place)
    fly = InstantaneousAction("fly", a=Place, b=Place)
    for act, link in ((drive, road), (fly, highway)):
        a, b = act.parameters
        act.add_precondition(at(a))
        act.add_precondition(link(a, b))
        act.add_effect(at(a), False)
        act.add_effect(at(b), True)
    p = Problem("routes")
    for f in (at, road, highway):
        p.add_fluent(f, default_initial_value=False)
    p.add_actions([drive, fly])
    ps = [Object(n, Place) for n in "abcdefg"]
    p.add_objects(ps)
    p.set_initial_value(at(ps[0]), True)
    p.set_initial_value(highway(ps[0], ps[-1]), True)
    for x, y in zip(ps, ps[1:]):
        p.set_initial_value(road(x, y), True)
    p.add_goal(at(ps[-1]))
    p.add_quality_metric(MinimizeActionCosts({drive: 1, fly: 10}))
    return p


def test_anytime_improves_up_to_the_proven_optimum():
    p = routes()
    with AnytimePlanner(name="kairos") as planner:
        results = list(planner.get_solutions(p, timeout=10))
    assert len(results) >= 2
    assert all(r.status == S.INTERMEDIATE for r in results[:-1])
    assert results[-1].status == S.SOLVED_OPTIMALLY
    assert len(results[-1].plan.actions) == 6
    # As a oneshot planner with a metric and a timeout: improving too
    assert len(solve(routes()).plan.actions) == 6


def test_anytime_stopped_early():
    p = routes()
    with AnytimePlanner(name="kairos") as planner:
        for r in planner.get_solutions(p, timeout=30):
            assert r.plan is not None
            break  # leaving stops the search


# --- What UPF's writer leaves out ----------------------------------------------------------------

def test_object_fluents():
    Loc, Robot = UserType("Loc"), UserType("Robot")
    pos = Fluent("pos", Loc, r=Robot)
    go = InstantaneousAction("go", r=Robot, a=Loc, b=Loc)
    r, a, b = go.parameters
    go.add_precondition(up.shortcuts.Equals(pos(r), a))
    go.add_effect(pos(r), b)
    p = Problem("objects")
    p.add_fluent(pos)
    p.add_action(go)
    l1, l2, rob = Object("l1", Loc), Object("l2", Loc), Object("rob", Robot)
    p.add_objects([l1, l2, rob])
    p.set_initial_value(pos(rob), l1)
    p.add_goal(up.shortcuts.Equals(pos(rob), l2))
    result = solve(p)
    assert result.status == S.SOLVED_SATISFICING
    with PlanValidator(name="sequential_plan_validator") as v:
        assert v.validate(p, result.plan).status == V.VALID


def test_oversubscription_takes_the_best_gain():
    # Two rooms at the ends of a corridor, one robot: the farther one is worth more
    p, _, rs = robots(rooms=5)
    p.clear_goals()
    Room = rs[0].type
    at = p.fluent("at")
    back = InstantaneousAction("back", a=Room, b=Room)
    a, b = back.parameters
    back.add_precondition(at(a))
    back.add_precondition(p.fluent("next_to")(b, a))
    back.add_effect(at(a), False)
    back.add_effect(at(b), True)
    p.add_action(back)
    p.set_initial_value(at(rs[0]), False)
    p.set_initial_value(at(rs[2]), True)
    p.add_quality_metric(Oversubscription({at(rs[0]): 3, at(rs[4]): 5}))
    r = solve(p)
    assert r.status == S.SOLVED_OPTIMALLY
    assert r.plan.actions[-1].actual_parameters[1].object() == rs[4]
    with PlanValidator(name="kairos") as kv:
        v = kv.validate(p, r.plan)
    assert v.status == V.VALID
    assert v.metric_evaluations[p.quality_metrics[0]] == 5  # the gain, as UPF counts it


def test_bounded_fluents_are_kept():
    # A battery in [0, 100]: each move costs 40, so three moves cannot be done
    p, move, rs = robots(rooms=4)
    battery = Fluent("battery", RealType(0, 100))
    p.add_fluent(battery, default_initial_value=100)
    move.add_effect(battery, battery - 40)
    assert solve(p).status == S.UNSOLVABLE_PROVEN
    p2, move2, _ = robots(rooms=3)
    battery2 = Fluent("battery", IntType(0, 100))
    p2.add_fluent(battery2, default_initial_value=100)
    move2.add_effect(battery2, battery2 - 40)
    r = solve(p2)
    assert r.status == S.SOLVED_SATISFICING
    with PlanValidator(name="sequential_plan_validator") as v:
        assert v.validate(p2, r.plan).status == V.VALID


def test_timed_increases_are_unsupported():
    p, _, _ = robots()
    counter = Fluent("counter", IntType())
    p.add_fluent(counter, default_initial_value=0)
    p.add_increase_effect(up.model.timing.GlobalStartTiming(5), counter, 1)
    assert solve(p).status == S.UNSUPPORTED_PROBLEM


# --- Temporal ------------------------------------------------------------------------------------

def cellar(mend_takes=5):
    Match, Fuse = UserType("Match"), UserType("Fuse")
    unused = Fluent("unused", BoolType(), m=Match)
    light = Fluent("light", BoolType(), m=Match)
    mended = Fluent("mended", BoolType(), f=Fuse)
    handfree = Fluent("handfree", BoolType())
    lm = DurativeAction("light_match", m=Match)
    m = lm.parameter("m")
    lm.set_fixed_duration(8)
    lm.add_condition(StartTiming(), unused(m))
    lm.add_effect(StartTiming(), unused(m), False)
    lm.add_effect(StartTiming(), light(m), True)
    lm.add_effect(EndTiming(), light(m), False)
    mend = DurativeAction("mend", f=Fuse, m=Match)
    f, m2 = mend.parameters
    mend.set_fixed_duration(mend_takes)
    mend.add_condition(StartTiming(), handfree)
    mend.add_condition(up.model.ClosedTimeInterval(StartTiming(), EndTiming()), light(m2))
    mend.add_effect(StartTiming(), handfree, False)
    mend.add_effect(EndTiming(), handfree, True)
    mend.add_effect(EndTiming(), mended(f), True)
    p = Problem("cellar")
    for fl in (unused, light, mended, handfree):
        p.add_fluent(fl, default_initial_value=False)
    p.add_actions([lm, mend])
    m1, f1 = Object("m1", Match), Object("f1", Fuse)
    p.add_objects([m1, f1])
    p.set_initial_value(unused(m1), True)
    p.set_initial_value(handfree, True)
    p.add_goal(mended(f1))
    return p


def test_required_concurrency():
    p = cellar()
    r = solve(p)
    assert r.status == S.SOLVED_SATISFICING
    assert isinstance(r.plan, up.plans.TimeTriggeredPlan)
    with PlanValidator(name="up_time_triggered_validator") as v:
        assert v.validate(p, r.plan).status == V.VALID
    with PlanValidator(name="kairos") as kv:
        assert kv.validate(p, r.plan).status == V.VALID


def test_validator_says_why():
    p = cellar()
    lm, mend = p.action("light_match"), p.action("mend")
    m1, f1 = p.object("m1"), p.object("f1")
    late = up.plans.TimeTriggeredPlan([
        (Fraction(0), up.plans.ActionInstance(lm, [m1]), Fraction(8)),
        (Fraction(4), up.plans.ActionInstance(mend, [f1, m1]), Fraction(5))])
    with PlanValidator(name="kairos") as kv:
        v = kv.validate(p, late)
    assert v.status == V.INVALID
    assert v.log_messages


# --- Through UPF's meta engines ------------------------------------------------------------------

def test_replanner():
    p, _, rs = robots(rooms=4)
    with Replanner(problem=p, name="replanner[kairos]") as replanner:
        assert len(replanner.resolve(timeout=10).plan.actions) == 3
        replanner.remove_goal(p.goals[0])
        replanner.add_goal(p.fluent("at")(rs[1]))
        assert len(replanner.resolve(timeout=10).plan.actions) == 1


def test_credits():
    credits = up_kairos.KairosEngine.get_credits()
    assert credits.name == "Kairos"
    assert "Apache" in credits.license
