import argparse
import os
import shutil
from dataclasses import dataclass
from typing import List, Tuple, Optional
from urllib.parse import quote_plus

from PIL import Image
from sqlalchemy import create_engine, text

# 复用当前项目里已验证的 128 维特征提取器（与 attraction_images.embedding vector(128) 对齐）
from src.cv.feature_extractor import get_feature_extractor


@dataclass
class DBConfig:
    host: str = "localhost"
    user: str = "zhangbi_user"
    password: str = "Zhangbi@2025!"
    dbname: str = "attractions_db"
    port: int = 5432

    @property
    def sqlalchemy_url(self) -> str:
        user_enc = quote_plus(self.user)
        pwd_enc = quote_plus(self.password)
        return (
            f"postgresql+psycopg2://{user_enc}:{pwd_enc}"
            f"@{self.host}:{self.port}/{self.dbname}"
        )


def to_pgvector_literal(vec: List[float]) -> str:
    # '[0.1,0.2,...]'
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"


def ensure_image_exists(path: str) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"图片不存在: {path}")


def ensure_attraction_exists(conn, attraction_id: int) -> str:
    row = conn.execute(
        text("SELECT id, name FROM attractions WHERE id = :aid"),
        {"aid": attraction_id},
    ).fetchone()
    if not row:
        raise ValueError(f"attractions 中不存在 id={attraction_id} 的景点")
    return row.name


def extract_embedding(image_path: str) -> List[float]:
    extractor = get_feature_extractor()
    with Image.open(image_path).convert("RGB") as img:
        emb = extractor.extract(img)
    if not emb:
        raise RuntimeError("特征提取失败，embedding 为空")
    if len(emb) != 128:
        raise RuntimeError(f"embedding维度异常: {len(emb)}，期望128")
    return emb


def insert_attraction_image(
    conn,
    attraction_id: int,
    file_path: str,
    upload_by: str,
    embedding: List[float],
) -> int:
    emb_literal = to_pgvector_literal(embedding)
    row = conn.execute(
        text(
            """
            INSERT INTO attraction_images (attraction_id, file_path, upload_by, embedding)
            VALUES (:attraction_id, :file_path, :upload_by, CAST(:embedding AS vector))
            RETURNING id
            """
        ),
        {
            "attraction_id": attraction_id,
            "file_path": file_path,
            "upload_by": upload_by,
            "embedding": emb_literal,
        },
    ).fetchone()
    return int(row.id)


def search_similar_images(
    conn,
    query_embedding: List[float],
    top_k: int = 5,
    exclude_image_id: Optional[int] = None,
) -> List[Tuple]:
    emb_literal = to_pgvector_literal(query_embedding)

    sql = (
        """
        SELECT
            ai.id,
            ai.attraction_id,
            a.name AS attraction_name,
            ai.file_path,
            ai.upload_by,
            (ai.embedding <-> CAST(:embedding AS vector)) AS l2_distance
        FROM attraction_images ai
        JOIN attractions a ON a.id = ai.attraction_id
        WHERE ai.embedding IS NOT NULL
        """
    )

    params = {"embedding": emb_literal, "top_k": top_k}
    if exclude_image_id is not None:
        sql += " AND ai.id <> :exclude_image_id "
        params["exclude_image_id"] = exclude_image_id

    sql += " ORDER BY ai.embedding <-> CAST(:embedding AS vector) ASC LIMIT :top_k "

    rows = conn.execute(text(sql), params).fetchall()
    return rows


def save_uploaded_copy(src_path: str, save_dir: str) -> str:
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.basename(src_path)
    dst = os.path.join(save_dir, filename)

    # 避免重名覆盖
    if os.path.exists(dst):
        stem, ext = os.path.splitext(filename)
        i = 1
        while True:
            candidate = os.path.join(save_dir, f"{stem}_{i}{ext}")
            if not os.path.exists(candidate):
                dst = candidate
                break
            i += 1

    shutil.copy2(src_path, dst)
    return dst


def cmd_add_and_search(args):
    cfg = DBConfig(
        host=args.db_host,
        user=args.db_user,
        password=args.db_password,
        dbname=args.db_name,
        port=args.db_port,
    )

    ensure_image_exists(args.image_path)

    engine = create_engine(cfg.sqlalchemy_url, future=True)

    with engine.begin() as conn:
        attraction_name = ensure_attraction_exists(conn, args.attraction_id)

        # 1) 保存图片副本（仅在当前项目目录下）
        stored_path = save_uploaded_copy(args.image_path, args.storage_dir)

        # 2) 提特征
        embedding = extract_embedding(stored_path)

        # 3) 入库 attraction_images
        new_image_id = insert_attraction_image(
            conn=conn,
            attraction_id=args.attraction_id,
            file_path=stored_path,
            upload_by=args.upload_by,
            embedding=embedding,
        )

        # 4) 以新图作为“标准图”做相似检索
        similar = search_similar_images(
            conn=conn,
            query_embedding=embedding,
            top_k=args.top_k,
            exclude_image_id=new_image_id,
        )

    print("\n✅ 新图入库成功")
    print(f"- 景点: {attraction_name} (id={args.attraction_id})")
    print(f"- 新图ID: {new_image_id}")
    print(f"- 存储路径: {stored_path}")
    print("\n🔍 相似检索结果:")
    if not similar:
        print("- 无结果（可能库里暂无其他带embedding的图片）")
        return

    for i, row in enumerate(similar, start=1):
        print(
            f"{i}. image_id={row.id}, attraction_id={row.attraction_id}, "
            f"attraction={row.attraction_name}, distance={row.l2_distance:.6f}, path={row.file_path}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="在 attraction_images 中插入新图片并以该图做相似度检索（128维）"
    )

    parser.add_argument("--image-path", required=True, help="要上传的新图片路径")
    parser.add_argument("--attraction-id", type=int, required=True, help="归属景点ID")
    parser.add_argument("--upload-by", default="multimodal_cli", help="上传人")
    parser.add_argument("--top-k", type=int, default=5, help="返回相似结果数量")

    parser.add_argument(
        "--storage-dir",
        default="/home/zhangbi/Zhangbi_Traveler/DataBase/Search_Update_Context/json/pgvector_optimized/src/multimodal/uploaded_images",
        help="图片复制存储目录（建议保持在multimodal目录内）",
    )

    # DB 参数（默认就是你提供的库）
    parser.add_argument("--db-host", default="localhost")
    parser.add_argument("--db-port", type=int, default=5432)
    parser.add_argument("--db-user", default="zhangbi_user")
    parser.add_argument("--db-password", default="Zhangbi@2025!")
    parser.add_argument("--db-name", default="attractions_db")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    cmd_add_and_search(args)


if __name__ == "__main__":
    main()
