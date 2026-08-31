"""正式节点与图片资产绑定的前置校验。

该模块只做无副作用的数据校验，供结构化同步在写入 node_assets 前调用。
它不参与节点补全、候选抽取或自增长抽取。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def validate_asset_binding_target(
    binding: Mapping[str, Any],
    existing_node_ids: Iterable[Any],
) -> dict[str, Any]:
    """校验图片绑定是否指向当前景区中已存在的正式节点。

    输入：
        binding：A 端上传的图片绑定对象，至少应包含 source_asset_id、
            source_node_id，object_type 可省略或为 node。
        existing_node_ids：本次同步后数据库中该景区的 source_node_id 集合。

    输出：
        包含 allowed、reason、asset_id、target_node_id 的字典。
        allowed=True 才允许写入 node_assets；失败原因使用稳定编码，
        便于同步任务诊断和页面展示。
    """
    object_type = binding.get("object_type")
    asset_id = str(binding.get("source_asset_id") or "").strip()
    target_node_id = str(binding.get("source_node_id") or "").strip()
    existing = {
        str(node_id).strip()
        for node_id in existing_node_ids
        if node_id is not None and str(node_id).strip()
    }

    if object_type not in (None, "node"):
        reason = "UNSUPPORTED_OBJECT_TYPE"
    elif not asset_id:
        reason = "ASSET_ID_REQUIRED"
    elif not target_node_id:
        reason = "TARGET_NODE_ID_REQUIRED"
    elif target_node_id not in existing:
        reason = "TARGET_NODE_NOT_FOUND"
    else:
        reason = "OK"

    return {
        "allowed": reason == "OK",
        "reason": reason,
        "asset_id": asset_id or None,
        "target_node_id": target_node_id or None,
    }
