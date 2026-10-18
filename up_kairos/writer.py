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

"""UPF problems in the PDDL Kairos reads: UPF's writer, plus what Kairos plans with and the
writer leaves out (object fluents, oversubscription, bounded numeric fluents)."""

from fractions import Fraction
from io import StringIO

import unified_planning as up
from unified_planning.io import PDDLWriter
from unified_planning.io.pddl_writer import ConverterToPDDLString
from unified_planning.model import Fluent, Variable


def _bounded(problem):
    """The problem with the bounds of its numeric fluents as state invariants (PDDL has no
    bounded types: UPF's writer drops them), or the problem itself if none has bounds."""
    em = problem.environment.expression_manager
    invariants = []
    for f in problem.fluents:
        t = f.type
        if not (t.is_int_type() or t.is_real_type()):
            continue
        if t.lower_bound is None and t.upper_bound is None:
            continue
        params = [Variable(p.name, p.type, problem.environment) for p in f.signature]
        value = f(*params)
        bounds = []
        if t.lower_bound is not None:
            bounds.append(em.GE(value, t.lower_bound))
        if t.upper_bound is not None:
            bounds.append(em.LE(value, t.upper_bound))
        body = em.And(bounds)
        invariants.append(em.Forall(body, *params) if params else body)
    if not invariants:
        return problem
    bounded = problem.clone()
    for inv in invariants:
        bounded.add_state_invariant(inv)
    return bounded


class _Problem:
    """The problem as the writer must see it: some attributes replaced, the rest its own."""

    def __init__(self, problem, **replaced):
        self._problem = problem
        self._replaced = replaced

    def __getattr__(self, name):
        if name in self._replaced:
            return self._replaced[name]
        return getattr(self._problem, name)


def _oversubscription(problem):
    """The oversubscription metric of the problem, if it has one."""
    for m in problem.quality_metrics:
        if m.is_oversubscription():
            return m
    return None


def _number(x) -> str:
    x = Fraction(x)
    return str(x.numerator) if x.denominator == 1 else repr(float(x))


class KairosWriter(PDDLWriter):
    """UPF's PDDL writer, with the requirements and boolean assignments as conditional
    effects, and:

    - object fluents, as PDDL 3.1 (``(:functions (f ?x - t) - type)``);
    - oversubscription, as preferences on the goals and a metric of the gain lost;
    - bounded numeric fluents, as state invariants.

    Names map back to the original problem.
    """

    def __init__(self, problem):
        self._original = problem
        super().__init__(_bounded(problem), needs_requirements=True,
                         rewrite_bool_assignments=True)
        for effects in problem.timed_effects.values():
            if any(e.is_increase() or e.is_decrease() for e in effects):
                raise up.exceptions.UPUnsupportedProblemTypeError(
                    "timed effects that increase or decrease: Kairos reads timed initial "
                    "fluents as assignments")
        self._gains = None
        metric = _oversubscription(problem)
        if metric is not None:
            if len(problem.quality_metrics) > 1:
                raise up.exceptions.UPUnsupportedProblemTypeError(
                    "Kairos takes one quality metric")
            self._gains = list(metric.goals.items())

    @property
    def total_gain(self):
        """Oversubscription: the gain of every optional goal (Kairos' metric is what is lost)."""
        return sum(Fraction(g) for _, g in self._gains) if self._gains else None

    def get_item_named(self, name):
        item = super().get_item_named(name)
        if self.problem is self._original:
            return item
        # The clone with invariants has the same names: its items, in the original problem
        if self._original.has_action(item.name):
            return self._original.action(item.name)
        if self._original.has_object(item.name):
            return self._original.object(item.name)
        return item

    # --- Domain ----------------------------------------------------------------------------------

    def _write_domain(self, out):
        objects = [f for f in self.problem.fluents if f.type.is_user_type()]
        if not objects and self._gains is None:
            return super()._write_domain(out)
        # Object fluents: numeric stand-ins with the same PDDL names while UPF writes, then
        # their types in the declarations
        stand_ins = {}
        for f in objects:
            name = self._get_mangled_name(f)
            s = Fluent(f.name, self.problem.environment.type_manager.IntType(), f.signature,
                       self.problem.environment)
            self.otn_renamings[s] = name
            stand_ins[f] = s
        text = self._written(super()._write_domain, fluents=[stand_ins.get(f, f)
                                                            for f in self.problem.fluents])
        for f in objects:
            params = "".join(f" {self._get_mangled_name(p)} - {self._get_mangled_name(p.type)}"
                             for p in f.signature)
            declaration = f"({self._get_mangled_name(f)}{params})"
            at = text.index(declaration, text.index("(:functions"))
            end = at + len(declaration)
            text = text[:end] + f" - {self._get_mangled_name(f.type)}" + text[end:]
        extra = (" :object-fluents" if objects else "") + \
            (" :preferences" if self._gains is not None else "")
        text = text.replace("(:requirements :strips", "(:requirements :strips" + extra, 1)
        out.write(text)

    def _written(self, write, **replaced):
        """What `write` writes with the problem as _Problem(problem, **replaced)."""
        buf = StringIO()
        real = self.problem
        self.problem = _Problem(real, **replaced)
        try:
            write(buf)
        finally:
            self.problem = real
        return buf.getvalue()

    # --- Problem ---------------------------------------------------------------------------------

    def _write_problem(self, out):
        if self._gains is None:
            return super()._write_problem(out)
        text = self._written(super()._write_problem, quality_metrics=[])
        converter = ConverterToPDDLString(self.problem.environment, self._get_mangled_name)
        preferences, terms = [], []
        for i, (goal, gain) in enumerate(self._gains):
            name = f"kairos-gain-{i}"
            preferences.append(f"\n           (preference {name} {converter.convert(goal)})")
            terms.append(f"(* {_number(gain)} (is-violated {name}))")
        at = text.index("\n        )\n )\n", text.index(" (:goal (and"))
        text = text[:at] + "".join(preferences) + text[at:]
        lost = terms[0] if len(terms) == 1 else f"(+ {' '.join(terms)})"
        end = text.rstrip().rindex(")")
        text = text[:end] + f" (:metric minimize {lost})\n" + text[end:]
        out.write(text)
