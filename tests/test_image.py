import os
import sys
from sqlalchemy import text

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.database.session import SessionLocal
from src.cv.feature_extractor import FeatureExtractor

MODEL_PATH = "/home/zhangbi/Zhangbi_Traveler/DataBase/Search_Update_Context/json/pgvector/models/simclr_checkpoints/simclr_zhangbi_final.pth"
TEST_IMAGE_PATH = "/home/zhangbi/Zhangbi_Traveler/DataBase/Search_Update_Context/json/figure/images_webp/baoqiang_1.webp" 

def test_search():
    db = SessionLocal()
    
    # 1. 加载模型
    try:
        extractor = FeatureExtractor(MODEL_PATH)
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # 2. 提取测试图片的特征
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"❌ 测试图片不存在: {TEST_IMAGE_PATH}") 
        return
        
    print(f"🔍 正在提取特征: {TEST_IMAGE_PATH}")
    query_vector = extractor.extract(TEST_IMAGE_PATH)
    
    if not query_vector:
        print("❌ 特征提取失败")
        return

    # 3. 在数据库中搜索最相似的图片
    # 使用 pgvector 的 <-> 操作符 (欧氏距离) 或 <=> (余弦距离)
    # SimCLR 特征通常归一化过，所以余弦距离和欧氏距离效果差不多，这里用 <->
    print("🚀 正在数据库中搜索最相似的景点...")
    
    sql = text("""
        SELECT 
            ai.id, 
            ai.file_path, 
            a.name as attraction_name,
            (ai.embedding <-> :query_vector) as distance
        FROM attraction_images ai
        JOIN attractions a ON ai.attraction_id = a.id
        ORDER BY distance ASC
        LIMIT 3;
    """)
    
    results = db.execute(sql, {"query_vector": str(query_vector)}).fetchall()
    
    # 4. 打印结果
    print("\n🏆 搜索结果 (Top 3):")
    print("-" * 50)
    for idx, row in enumerate(results):
        print(f"第 {idx+1} 名:")
        print(f"  景点名称: {row.attraction_name}")
        print(f"  图片路径: {row.file_path}")
        print(f"  距离差异: {row.distance:.4f} (越小越好)")
        print("-" * 50)

    db.close()

if __name__ == "__main__":
    test_search()