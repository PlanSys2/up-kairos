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

"""Kairos as an engine of the Unified Planning Framework.

    import up_kairos
    up_kairos.register()            # until UPF lists it among its engines
    with OneshotPlanner(name="kairos") as planner:
        result = planner.solve(problem, timeout=10)
"""

from .engine import KairosEngine, KairosValidator

__version__ = "0.2.0"
__all__ = ["KairosEngine", "KairosValidator", "register"]


def register(environment=None):
    """Adds Kairos to the engines of the UPF environment (the global one by default)."""
    from unified_planning.environment import get_environment
    env = environment if environment is not None else get_environment()
    if "kairos" not in env.factory.engines:
        env.factory.add_engine("kairos", __name__, "KairosEngine")
    if "kairos-val" not in env.factory.engines:
        env.factory.add_engine("kairos-val", __name__, "KairosValidator")
