import os
import sys

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.database.session import SessionLocal
from src.database.models_existing import AttractionImage
from src.cv.feature_extractor import FeatureExtractor


MODEL_PATH = "/home/zhangbi/Zhangbi_Traveler/DataBase/Search_Update_Context/json/pgvector/models/simclr_checkpoints/simclr_zhangbi_final.pth"


IMAGE_BASE_DIR = "/home/zhangbi/Zhangbi_Traveler/DataBase/Search_Update_Context/json/figure"

def update_embeddings():
    db = SessionLocal()
    
    try:
        extractor = FeatureExtractor(MODEL_PATH)
        print("✅ 模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    print("🔍 查询待处理图片...")
    images = db.query(AttractionImage).filter(AttractionImage.embedding == None).all()
    
    total = len(images)
    print(f"📝 待处理数量: {total}")

    count = 0
    for img in images:
        # 拼接完整路径
        # 数据库: images_webp/baoqiang_1.webp
        # 根目录: .../json/figure
        # 结果: .../json/figure/images_webp/baoqiang_1.webp
        full_path = os.path.join(IMAGE_BASE_DIR, img.file_path)

        if not os.path.exists(full_path):
            print(f"⚠️ 文件不存在: {full_path}")
            continue
            
        # 提取特征
        vector = extractor.extract(full_path)
        
        if vector:
            img.embedding = vector
            count += 1
            print(f"[{count}/{total}] 已提取: {os.path.basename(full_path)}")
            
            if count % 10 == 0:
                db.commit()
        else:
            print(f"❌ 提取失败: {full_path}")

    db.commit()
    db.close()
    print("🎉 完成！")

if __name__ == "__main__":
    update_embeddings()