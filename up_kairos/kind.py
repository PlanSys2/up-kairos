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

"""What Kairos plans with, as UPF problem kinds (each one tested: tests/test_kinds.py)."""

from unified_planning.model import ProblemKind
from unified_planning.model.problem_kind_versioning import LATEST_PROBLEM_KIND_VERSION

FEATURES = {
    "PROBLEM_CLASS": ["ACTION_BASED"],
    "PROBLEM_TYPE": ["SIMPLE_NUMERIC_PLANNING", "GENERAL_NUMERIC_PLANNING"],
    # Not SELF_OVERLAPPING (one instance of a ground durative action at a time, as POPF),
    # nor intermediate conditions or timed goals (no PDDL for them)
    "TIME": ["CONTINUOUS_TIME", "TIMED_EFFECTS", "DURATION_INEQUALITIES", "PROCESSES",
             "EVENTS"],
    "EXPRESSION_DURATION": ["STATIC_FLUENTS_IN_DURATIONS", "FLUENTS_IN_DURATIONS",
                            "INT_TYPE_DURATIONS", "REAL_TYPE_DURATIONS"],
    "CONDITIONS_KIND": ["NEGATIVE_CONDITIONS", "DISJUNCTIVE_CONDITIONS", "EQUALITIES",
                        "EXISTENTIAL_CONDITIONS", "UNIVERSAL_CONDITIONS"],
    "EFFECTS_KIND": ["CONDITIONAL_EFFECTS", "INCREASE_EFFECTS", "DECREASE_EFFECTS",
                     "INCREASE_CONTINUOUS_EFFECTS", "DECREASE_CONTINUOUS_EFFECTS",
                     "NON_LINEAR_CONTINUOUS_EFFECTS", "STATIC_FLUENTS_IN_BOOLEAN_ASSIGNMENTS",
                     "STATIC_FLUENTS_IN_NUMERIC_ASSIGNMENTS", "FLUENTS_IN_BOOLEAN_ASSIGNMENTS",
                     "FLUENTS_IN_NUMERIC_ASSIGNMENTS", "FORALL_EFFECTS",
                     "STATIC_FLUENTS_IN_OBJECT_ASSIGNMENTS", "FLUENTS_IN_OBJECT_ASSIGNMENTS"],
    "TYPING": ["FLAT_TYPING", "HIERARCHICAL_TYPING"],
    "FLUENTS_TYPE": ["INT_FLUENTS", "REAL_FLUENTS", "OBJECT_FLUENTS"],  # PDDL 3.1
    # Oversubscription as preferences on the goals (writer.py)
    "QUALITY_METRICS": ["ACTIONS_COST", "FINAL_VALUE", "MAKESPAN", "PLAN_LENGTH",
                        "OVERSUBSCRIPTION"],
    "OVERSUBSCRIPTION_KIND": ["INT_NUMBERS_IN_OVERSUBSCRIPTION",
                              "REAL_NUMBERS_IN_OVERSUBSCRIPTION"],
    "ACTIONS_COST_KIND": ["STATIC_FLUENTS_IN_ACTIONS_COST", "FLUENTS_IN_ACTIONS_COST",
                          "INT_NUMBERS_IN_ACTIONS_COST", "REAL_NUMBERS_IN_ACTIONS_COST"],
    "CONSTRAINTS_KIND": ["TRAJECTORY_CONSTRAINTS", "STATE_INVARIANTS"],
    "NUMBERS": ["BOUNDED_TYPES"],  # as state invariants (writer.py)
    "INITIAL_STATE": ["UNDEFINED_INITIAL_NUMERIC"],
}


def _kind() -> ProblemKind:
    kind = ProblemKind(version=LATEST_PROBLEM_KIND_VERSION)
    for group, features in FEATURES.items():
        for f in features:
            getattr(kind, "set_" + group.lower())(f)
    return kind


SUPPORTED_KIND = _kind()
