import sys
import os

# 将项目目录加入 Python 路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.database.base import Base
from src.database.session import engine
from src.database.models import * # 确保所有模型都被加载

def init_db():
    print("--- 正在远程数据库上初始化表结构 ---")
    try:
        # 这会自动在远程数据库创建所有 Base 派生的表
        Base.metadata.create_all(bind=engine)
        print("✅ 所有表已成功创建或已存在。")
    except Exception as e:
        print(f"❌ 初始化失败: {e}")

if __name__ == "__main__":
    init_db()
