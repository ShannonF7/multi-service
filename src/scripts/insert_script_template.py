"""
读取一个 JSON 文件并将其作为一条 `ScriptTemplate` 记录插入数据库。

用法示例：
python -m src.scripts.insert_script_template --file "/path/to/scripts_template.json" \
    --created-by importer_name --suitable-people 4
"""

import argparse
import json
import os
import re
import sys
import uuid
from typing import Any, Dict, Optional

from sqlalchemy.exc import SQLAlchemyError

try:
    from src.database.session import SessionLocal
    from src.database.models import ScriptTemplate
except Exception as e:
    print("无法导入项目的数据库模块，请确保在项目根目录运行此脚本。错误：", e)
    raise


def extract_suitable_people(recommended: Optional[str]) -> Optional[int]:
    if not recommended:
        return None
    m = re.search(r"(\d+)", recommended)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    return None


def build_record_from_json(json_data: Dict[str, Any], created_by: str, suitable_override: Optional[int]) -> ScriptTemplate:
    metadata = json_data.get("system_config", {}).get("script_metadata", {})

    name = "张壁：星野藏甲(最终版)"
    style = metadata.get("style") or "unknown"

    if suitable_override is not None:
        suitable_people = suitable_override
    else:
        rec = metadata.get("recommended_group_size")
        suitable_people = extract_suitable_people(rec) or 1

    st = ScriptTemplate(
        id=uuid.uuid4(),
        name=name,
        style=style,
        suitable_people=suitable_people,
        template=json_data,
        created_by=created_by,
        is_active=True,
    )
    return st


def main():
    parser = argparse.ArgumentParser(description="Insert a JSON script as a ScriptTemplate record.")
    parser.add_argument("--file", "-f", required=True, help="Path to the JSON file to import")
    parser.add_argument("--created-by", "-c", default="import_script", help="Value for created_by column")
    parser.add_argument("--suitable-people", "-s", type=int, default=None, help="Override for suitable_people")
    args = parser.parse_args()

    json_path = os.path.abspath(args.file)
    if not os.path.exists(json_path):
        print(f"指定的文件不存在: {json_path}")
        sys.exit(2)

    with open(json_path, "r", encoding="utf-8") as fh:
        try:
            data = json.load(fh)
        except json.JSONDecodeError as e:
            print("无法解析 JSON：", e)
            sys.exit(3)

    session = SessionLocal()
    try:
        record = build_record_from_json(data, created_by=args.created_by, suitable_override=args.suitable_people)
        session.add(record)
        session.commit()
        session.refresh(record)
        print("插入成功，记录 id：", getattr(record, "id", None))
    except SQLAlchemyError as e:
        session.rollback()
        print("数据库错误：", e)
        raise
    finally:
        session.close()


if __name__ == "__main__":
    main()
