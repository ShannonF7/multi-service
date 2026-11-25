"""P3-C 任务 Skill 注册与选择。

注册表只负责把 QuerySpec/ExecutionPlan 映射到任务 Skill；具体数据访问仍由
skill_router 中的 GraphTool、FactTool、TextTool、ImageTool 完成。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional


@dataclass(frozen=True)
class SkillRegistration:
    name: str
    skill: Any
    intents: frozenset[str]
    step_types: frozenset[str]
    request_types: frozenset[str]


class SkillRegistry:
    """任务 Skill 的显式注册表，避免在 Router 中散落 if/else 映射。"""

    def __init__(self) -> None:
        self._registrations: Dict[str, SkillRegistration] = {}

    def register(
        self,
        skill: Any,
        *,
        intents: Iterable[str],
        step_types: Iterable[str],
        request_types: Iterable[str] = (),
    ) -> Any:
        name = str(skill.name)
        if name in self._registrations:
            raise ValueError(f"任务 Skill 已注册：{name}")
        registration = SkillRegistration(
            name=name,
            skill=skill,
            intents=frozenset(str(item).upper() for item in intents),
            step_types=frozenset(str(item).upper() for item in step_types),
            request_types=frozenset(str(item).upper() for item in request_types),
        )
        self._registrations[name] = registration
        return skill

    def get(self, name: str) -> Optional[Any]:
        registration = self._registrations.get(str(name))
        return registration.skill if registration else None

    def names(self) -> List[str]:
        return list(self._registrations)

    def select(
        self,
        *,
        intent: str,
        step_type: str,
        request_types: Iterable[str] = (),
    ) -> Optional[Any]:
        """优先按具体任务请求选择，再按意图和步骤类型兜底。"""
        intent = str(intent or "UNKNOWN").upper()
        step_type = str(step_type or "").upper()
        request_types = {str(item).upper() for item in request_types}
        for registration in self._registrations.values():
            if (
                request_types
                and registration.request_types
                and request_types.intersection(registration.request_types)
                and step_type in registration.step_types
            ):
                return registration.skill
        for registration in self._registrations.values():
            if intent in registration.intents and step_type in registration.step_types:
                return registration.skill
        for registration in self._registrations.values():
            if step_type in registration.step_types:
                return registration.skill
        return None

    def select_for_plan(
        self,
        *,
        intent: str,
        steps: Iterable[Any],
        request_types: Iterable[str] = (),
    ) -> List[Any]:
        selected: List[Any] = []
        for step in steps:
            skill = self.select(
                intent=intent,
                step_type=getattr(step, "type", ""),
                request_types=request_types,
            )
            if skill is not None:
                selected.append(skill)
        if not selected and str(intent or "").upper() == "CASUAL":
            casual = self.get("CasualSkill")
            if casual is not None:
                selected.append(casual)
        return selected

