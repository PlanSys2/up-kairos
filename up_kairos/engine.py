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

"""Kairos as an engine of the Unified Planning Framework: planning in the same process
(kairos-planner), anytime with a proof of the optimum, and its validator."""

from fractions import Fraction
from typing import IO, Callable, Iterator, List, Optional

import kairos
import unified_planning as up
from unified_planning.engines import Credits, Engine
from unified_planning.engines.mixins.anytime_planner import (
    AnytimeGuarantee,
    AnytimePlannerMixin,
)
from unified_planning.engines.mixins.oneshot_planner import (
    OneshotPlannerMixin,
    OptimalityGuarantee,
)
from unified_planning.engines.mixins.plan_validator import PlanValidatorMixin
from unified_planning.engines.results import (
    FailedValidationReason,
    LogLevel,
    LogMessage,
    PlanGenerationResult,
    PlanGenerationResultStatus,
    ValidationResult,
    ValidationResultStatus,
    correct_plan_generation_result,
)

from .kind import SUPPORTED_KIND, VALIDATOR_KIND
from .writer import KairosWriter

credits = Credits(
    "Kairos",
    "Intelligent Robotics Lab, Universidad Rey Juan Carlos",
    "fmrico@gmail.com",
    "https://github.com/PlanSys2/kairos",
    "Apache-2.0",
    "Temporal and numeric PDDL planner (PDDL 2.1-3.1, PDDL+) with proofs of unsolvability.",
    "Kairos plans classical, numeric, temporal (required concurrency, timed initial literals "
    "and fluents, duration inequalities), PDDL3 (preferences, trajectory constraints) and "
    "PDDL+ (processes, events) tasks, anytime, and proves when no plan exists. Every plan is "
    "validated before it is returned.",
)

# Kairos' statuses without a plan, in UPF's terms
_NO_PLAN = {
    kairos.Status.UNSOLVABLE: PlanGenerationResultStatus.UNSOLVABLE_PROVEN,
    kairos.Status.NO_PLAN_FOUND: PlanGenerationResultStatus.UNSOLVABLE_INCOMPLETELY,
    kairos.Status.TIMEOUT: PlanGenerationResultStatus.TIMEOUT,
    kairos.Status.CANCELLED: PlanGenerationResultStatus.TIMEOUT,
    kairos.Status.OUT_OF_MEMORY: PlanGenerationResultStatus.MEMOUT,
    kairos.Status.GROUNDING_TOO_LARGE: PlanGenerationResultStatus.MEMOUT,
    kairos.Status.UNSUPPORTED: PlanGenerationResultStatus.UNSUPPORTED_PROBLEM,
    kairos.Status.INVALID_INPUT: PlanGenerationResultStatus.INTERNAL_ERROR,
    kairos.Status.INTERNAL_ERROR: PlanGenerationResultStatus.INTERNAL_ERROR,
}

_LOG_LEVEL = {
    kairos.Severity.ERROR: LogLevel.ERROR,
    kairos.Severity.WARNING: LogLevel.WARNING,
    kairos.Severity.NOTE: LogLevel.INFO,
}

# Options of KairosEngine(...) passed on to Kairos as they are
_KAIROS_OPTIONS = ("memory_limit_mb", "engine", "search", "threads", "seed",
                   "stop_after_no_improvement", "target_cost")


class KairosEngine(Engine, OneshotPlannerMixin, AnytimePlannerMixin):
    """Kairos in UPF (its validator is KairosValidator, "kairos-val").

    Options (``OneshotPlanner(name="kairos", params={...})``):

    - ``anytime``: keep improving until the timeout or a proof of the optimum. By default, as
      a oneshot planner, only when the problem has a quality metric and a timeout is given;
      always as an anytime planner.
    - ``timeout``: seconds when the call gives none (default 30).
    - ``threads`` (default 1), ``memory_limit_mb`` (0: half of what is available),
      ``engine`` ("auto", "m1", "m2", "m3"), ``search``, ``seed``,
      ``stop_after_no_improvement`` (seconds), ``target_cost``.
    """

    def __init__(self, **options):
        Engine.__init__(self)
        OneshotPlannerMixin.__init__(self)
        AnytimePlannerMixin.__init__(self)
        unknown = set(options) - set(_KAIROS_OPTIONS) - {"anytime", "timeout"}
        if unknown:
            raise up.exceptions.UPUsageError(f"unknown Kairos options: {sorted(unknown)}")
        self._options = options

    @property
    def name(self) -> str:
        return "kairos"

    @staticmethod
    def supported_kind() -> "up.model.ProblemKind":
        return SUPPORTED_KIND

    @staticmethod
    def supports(problem_kind: "up.model.ProblemKind") -> bool:
        return problem_kind <= KairosEngine.supported_kind()

    @staticmethod
    def satisfies(optimality_guarantee: OptimalityGuarantee) -> bool:
        # Satisficing; a plan proved optimal comes back as SOLVED_OPTIMALLY
        return optimality_guarantee == OptimalityGuarantee.SATISFICING

    @staticmethod
    def ensures(anytime_guarantee: AnytimeGuarantee) -> bool:
        return anytime_guarantee == AnytimeGuarantee.INCREASING_QUALITY

    @staticmethod
    def get_credits(**kwargs) -> Optional[Credits]:
        return credits

    # --- Translation -----------------------------------------------------------------------------

    def _task(self, problem):
        """The problem in PDDL, parsed by Kairos; the writer maps the names back."""
        writer = KairosWriter(problem)
        domain_text = writer.get_domain()
        problem_text = writer.get_problem()
        domain = kairos.parse_domain(domain_text, "domain.pddl")
        return writer, kairos.parse_problem(problem_text, domain, "problem.pddl")

    def _options_for(self, problem, timeout, anytime_default):
        o = kairos.SearchOptions()
        for name in _KAIROS_OPTIONS:
            if name in self._options:
                value = self._options[name]
                if name == "engine" and isinstance(value, str):
                    value = kairos.Engine[value.upper()]
                setattr(o, name, value)
        o.timeout = float(timeout if timeout is not None else self._options.get("timeout", 30))
        o.anytime = self._options.get("anytime", anytime_default)
        if problem.epsilon is not None:
            o.epsilon = float(problem.epsilon)
        return o

    def _plan_back(self, problem, writer, plan):
        """A Kairos plan as a UPF plan (names as the writer gave them)."""
        if plan is None:
            return None
        em = problem.environment.expression_manager

        def instance(step):
            action = writer.get_item_named(step.name)
            args = [em.ObjectExp(writer.get_item_named(a)) for a in step.args]
            return up.plans.ActionInstance(action, args)
        if not plan.temporal:
            return up.plans.SequentialPlan([instance(s) for s in plan.steps], problem.environment)
        timed = []
        for s in plan.steps:
            durative = isinstance(writer.get_item_named(s.name), up.model.DurativeAction)
            timed.append((_fraction(s.time), instance(s),
                          _fraction(s.duration) if durative else None))
        return up.plans.TimeTriggeredPlan(timed, problem.environment)

    def _logs(self, diagnostics) -> List[LogMessage]:
        return [LogMessage(_LOG_LEVEL[d.severity], str(d)) for d in diagnostics]

    @staticmethod
    def _metrics(stats) -> dict:
        return {
            "engine_internal_time": str(stats.total_ms / 1000.0),
            "kairos_expanded": str(stats.expanded),
            "kairos_states": str(stats.states),
            "kairos_plans": str(stats.plans),
            "kairos_peak_memory_mb": str(stats.peak_memory_bytes >> 20),
        }

    def _result(self, problem, writer, result) -> PlanGenerationResult:
        if result.plan is not None:
            optimal = result.stop_reason == kairos.StopReason.PROVEN_OPTIMAL and \
                problem.kind.has_quality_metrics()
            status = PlanGenerationResultStatus.SOLVED_OPTIMALLY if optimal else \
                PlanGenerationResultStatus.SOLVED_SATISFICING
        else:
            status = _NO_PLAN[result.status]
        metrics, logs = self._metrics(result.stats), self._logs(result.diagnostics)
        if result.plan is not None and result.plan.end is not None:
            # PDDL+: time passes after the last step; UPF's plans have no end of their own
            metrics["kairos_plan_end"] = str(_fraction(result.plan.end))
            logs.append(LogMessage(LogLevel.INFO, f"the plan ends at {result.plan.end} "
                                   "(processes and events act until then)"))
        res = PlanGenerationResult(
            status, self._plan_back(problem, writer, result.plan), self.name, metrics, logs)
        return self._corrected(problem, res)

    def _corrected(self, problem, res):
        if problem.kind.has_continuous_time() or problem.kind.has_discrete_time():
            if res.plan is None or isinstance(res.plan, up.plans.TimeTriggeredPlan):
                return correct_plan_generation_result(res, problem, self._epsilon(problem))
        return res

    def _epsilon(self, problem) -> Fraction:
        return Fraction(str(problem.epsilon)) if problem.epsilon is not None else Fraction(1, 1000)

    @staticmethod
    def _log_to(output_stream):
        if output_stream is None:
            return None
        return lambda level, channel, text: output_stream.write(f"[{channel.name.lower()}] {text}\n")

    def _failed(self, problem, error) -> PlanGenerationResult:
        """What Kairos could not read: unsupported (the writer) or a bug (the PDDL)."""
        status = PlanGenerationResultStatus.UNSUPPORTED_PROBLEM
        logs = [LogMessage(LogLevel.ERROR, str(error))]
        if isinstance(error, kairos.PddlError):
            status = PlanGenerationResultStatus.INTERNAL_ERROR
            logs = self._logs(error.diagnostics)
        return PlanGenerationResult(status, None, self.name, log_messages=logs)

    # --- OneshotPlanner --------------------------------------------------------------------------

    def _solve_with_params(self, problem, heuristic=None, timeout=None, output_stream=None,
                           warm_start_plan=None, **kwargs) -> PlanGenerationResult:
        if heuristic is not None:
            self._warn("Kairos does not use a custom heuristic")
        if warm_start_plan is not None:
            self._warn("Kairos does not start from a given plan")
        try:
            writer, task = self._task(problem)
        except (up.exceptions.UPException, kairos.PddlError) as e:
            return self._failed(problem, e)
        anytime = problem.kind.has_quality_metrics() and timeout is not None
        result = kairos.Planner().solve(task, self._options_for(problem, timeout, anytime),
                                        on_log=self._log_to(output_stream))
        return self._result(problem, writer, result)

    def _solve(self, problem, heuristic=None, timeout=None, output_stream=None):
        return self._solve_with_params(problem, heuristic, timeout, output_stream)

    # --- AnytimePlanner --------------------------------------------------------------------------

    def _get_solutions_with_params(self, problem, timeout=None, output_stream=None,
                                   warm_start_plan=None, **kwargs
                                   ) -> Iterator[PlanGenerationResult]:
        if warm_start_plan is not None:
            self._warn("Kairos does not start from a given plan")
        try:
            writer, task = self._task(problem)
        except (up.exceptions.UPException, kairos.PddlError) as e:
            yield self._failed(problem, e)
            return
        search = kairos.Planner()._planner.start(
            task, self._options_for(problem, timeout, True), None,
            self._log_to(output_stream))
        try:
            for plan in search.plans():
                yield self._corrected(problem, PlanGenerationResult(
                    PlanGenerationResultStatus.INTERMEDIATE,
                    self._plan_back(problem, writer, plan), self.name))
            yield self._result(problem, writer, search.wait())
        finally:
            search.stop()

    def _get_solutions(self, problem, timeout=None, output_stream=None):
        return self._get_solutions_with_params(problem, timeout, output_stream)

    def _warn(self, text):
        import warnings
        warnings.warn(text, stacklevel=3)


class KairosValidator(Engine, PlanValidatorMixin):
    """Kairos' validator in UPF ("kairos-val"): numeric, temporal, PDDL3, as VAL. Not for PDDL+
    (UPF's plans have no end time, which a PDDL+ plan needs)."""

    def __init__(self, **options):
        Engine.__init__(self)
        PlanValidatorMixin.__init__(self)
        if options:
            raise up.exceptions.UPUsageError(f"unknown Kairos options: {sorted(options)}")

    @property
    def name(self) -> str:
        return "kairos-val"

    @staticmethod
    def supported_kind() -> "up.model.ProblemKind":
        return VALIDATOR_KIND

    @staticmethod
    def supports(problem_kind: "up.model.ProblemKind") -> bool:
        return problem_kind <= KairosValidator.supported_kind()

    @staticmethod
    def supports_plan(plan_kind: "up.plans.PlanKind") -> bool:
        return plan_kind in (up.plans.PlanKind.SEQUENTIAL_PLAN,
                             up.plans.PlanKind.TIME_TRIGGERED_PLAN)

    @staticmethod
    def get_credits(**kwargs) -> Optional[Credits]:
        return credits

    # --- PlanValidator ---------------------------------------------------------------------------

    def _validate(self, problem, plan) -> ValidationResult:
        try:
            writer = KairosWriter(problem)
            domain_text, problem_text = writer.get_domain(), writer.get_problem()
            plan_text = writer.get_plan(plan)
            options = kairos.ValidateOptions()
            # Without an epsilon, UPF lets happenings at one instant interfere: Kairos' 0
            options.epsilon = float(problem.epsilon) if problem.epsilon is not None else 0.0
            report = kairos.validate_plan(domain_text, problem_text, kairos.parse_plan(plan_text),
                                          options)
        except (up.exceptions.UPException, kairos.PddlError) as e:
            return ValidationResult(ValidationResultStatus.UNKNOWN, self.name,
                                    [LogMessage(LogLevel.ERROR, str(e))])
        logs = [LogMessage(_LOG_LEVEL[d.severity], str(d)) for d in report.diagnostics]
        if report.status == kairos.ValidationStatus.VALID:
            metrics = None
            if report.metric is not None and len(problem.quality_metrics) == 1:
                value = Fraction(str(report.metric))
                if writer.total_gain is not None:
                    value = writer.total_gain - value  # Kairos minimizes the gain lost
                metrics = {problem.quality_metrics[0]: value}
            return ValidationResult(ValidationResultStatus.VALID, self.name, logs,
                                    metric_evaluations=metrics)
        if report.status != kairos.ValidationStatus.INVALID:
            return ValidationResult(ValidationResultStatus.UNKNOWN, self.name, logs)
        reason, inapplicable = FailedValidationReason.UNSATISFIED_GOALS, None
        if report.failed_step is not None:
            actions = _actions_of(plan)
            if report.failed_step < len(actions):
                inapplicable = actions[report.failed_step]
                reason = FailedValidationReason.INAPPLICABLE_ACTION
        if any(d.code == "validate.interference" for d in report.diagnostics):
            reason, inapplicable = FailedValidationReason.MUTEX_CONFLICT, None
        return ValidationResult(ValidationResultStatus.INVALID, self.name, logs,
                                reason=reason, inapplicable_action=inapplicable)


def _fraction(x: float) -> Fraction:
    """Times as Kairos prints them (to the nanosecond), exactly."""
    return Fraction(f"{x:.9f}")


def _actions_of(plan) -> List["up.plans.ActionInstance"]:
    """The action instances of a plan, in the order of its steps (sorted by time)."""
    if isinstance(plan, up.plans.SequentialPlan):
        return list(plan.actions)
    if isinstance(plan, up.plans.TimeTriggeredPlan):
        return [ai for _, ai, _ in sorted(plan.timed_actions, key=lambda x: x[0])]
    return []
