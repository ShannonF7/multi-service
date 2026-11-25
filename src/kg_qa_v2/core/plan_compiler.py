"""把 QuerySpec 编译成固定、可审计的 Graph/Text/Image 执行计划。"""

from __future__ import annotations

from ..models import ExecutionPlan, PlanStep, QuerySpec
from .entity_reference import is_image_deictic_reference


class PlanCompiler:
    def compile(self, spec: QuerySpec, *, max_evidence: int = 8) -> ExecutionPlan:
        if spec.route == "CASUAL":
            return ExecutionPlan(max_evidence=max_evidence)
        if spec.needs_clarification or spec.route == "CLARIFICATION":
            return ExecutionPlan(
                steps=[PlanStep(type="CLARIFICATION", reason="存在待确认对象，先完成用户选择")],
                max_evidence=max_evidence,
            )

        steps: list[PlanStep] = []
        target = spec.targets[0] if spec.targets else None
        resolved_ids = [
            str(item).strip()
            for item in (spec.constraints.get("resolved_node_ids") or [])
            if str(item).strip()
        ]
        resolved_node_id = resolved_ids[0] if resolved_ids else None
        needs_path = spec.intent == "PATH"
        request_types = {str(item.request_type).upper() for item in spec.requests}
        nearby_request = "NEARBY" in request_types
        visual_request = bool(request_types & {"VISUAL_APPEARANCE", "IMAGE_RETRIEVAL"})
        needs_image = (
            spec.intent == "IMAGE"
            or visual_request
            or bool(spec.response_requirements.images)
            or str(spec.constraints.get("response_modality") or "").upper() == "IMAGE"
        )
        has_uploaded_image = bool(str(spec.constraints.get("image_path") or "").strip())
        has_explicit_text_target = any(
            ref.reference_type == "EXPLICIT"
            and bool((ref.core_mention or ref.surface).strip())
            and not is_image_deictic_reference(ref)
            for ref in spec.entity_refs
        )
        # Uploaded-image queries without a named text target must not use the current GPS node.
        image_only_request = (
            spec.intent == "IMAGE"
            and has_uploaded_image
            and not has_explicit_text_target
        )
        needs_graph = (
            spec.route == "KNOWLEDGE"
            and spec.intent != "CASUAL"
            and not image_only_request
        )
        needs_text = spec.intent in {"EXPLAIN", "COMPARE"}
        attribute_requests = [item.model_dump() for item in spec.requests if item.request_type == "ATTRIBUTE"]
        if not attribute_requests and spec.attribute and spec.intent == "DIRECT_FACT":
            attribute_requests = [{"request_type": "ATTRIBUTE", "attribute": spec.attribute, "surface": ""}]
        if attribute_requests and resolved_node_id:
            steps.append(PlanStep(
                type="FACT",
                reason="属性问题通过已发布图谱的事实和关系查询回答",
                target=target,
                params={"operation": "facts", "node_id": resolved_node_id, "requests": attribute_requests},
            ))
        if nearby_request and resolved_node_id:
            nearby = next((item for item in spec.requests if item.request_type == "NEARBY"), None)
            steps.append(PlanStep(
                type="GRAPH",
                reason="周边问题通过已发布图谱坐标进行空间查询并按距离排序",
                target=target,
                params={
                    "operation": "nearby",
                    "node_id": resolved_node_id,
                    "radius_m": float(nearby.radius) if nearby and nearby.radius is not None else 300.0,
                    "category": nearby.category if nearby else None,
                    "limit": 20,
                },
            ))
        if needs_path:
            steps.append(PlanStep(
                type="GRAPH",
                reason="路径问题必须通过已发布图谱的路径查询回答",
                target=target,
                params={
                    "operation": "path",
                    "start": spec.start,
                    "end": spec.end,
                    "start_node_id": resolved_ids[0] if len(resolved_ids) > 0 else None,
                    "end_node_id": resolved_ids[1] if len(resolved_ids) > 1 else None,
                    "max_hops": 4,
                },
            ))
        elif needs_graph and not attribute_requests and not nearby_request:
            operation = "current_location" if spec.needs_current_location else (
                "neighborhood" if spec.intent in {"HIERARCHY", "LIST"} else "node_detail"
            )
            steps.append(PlanStep(
                type="GRAPH",
                reason=f"{spec.intent}问题优先读取当前领域已发布图谱",
                target=target,
                params={"operation": operation, "node_id": resolved_node_id},
            ))
        if needs_text:
            steps.append(PlanStep(
                type="TEXT",
                reason="补充领域文本证据，不替代已发布图谱实体确认",
                target=target,
                parallel_group="retrieval",
            ))
        if needs_image:
            steps.append(PlanStep(
                type="IMAGE",
                reason="用户要求或问题适合图文回答，检索已发布关联图片",
                target=target,
                parallel_group="retrieval",
                params={
                    "image_path": str(spec.constraints.get("image_path") or "").strip() or None,
                    "node_id": resolved_node_id if visual_request else None,
                    "image_query": target if visual_request else None,
                    "limit": 3,
                },
            ))
        return ExecutionPlan(
            steps=steps,
            parallel=sum(step.type in {"GRAPH", "FACT", "TEXT", "IMAGE"} for step in steps) > 1,
            max_hops=4 if spec.intent == "PATH" else 1,
            max_evidence=max_evidence,
        )
