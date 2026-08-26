from tabnanny import check
import torch
import numpy as np
from PIL import Image
import os
import time
import logging
import json
import requests
from torchvision import models, transforms

from openai import OpenAI
import mysql.connector

from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceBgeEmbeddings
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

import torch.nn.functional as F

# ----------------- MySQL 数据库连接 -----------------
class MySQLManager:
    def __init__(self, host, user, password, database):
        self.conn = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )
        self.cursor = self.conn.cursor()
        self._initialize_database()

    def _initialize_database(self):
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS spots (
            id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(255) NOT NULL
        );
        """
        )

        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS images (
            id INT AUTO_INCREMENT PRIMARY KEY,
            spot_id INT,
            image_path VARCHAR(1024),
            features BLOB,
            FOREIGN KEY (spot_id) REFERENCES spots(id)
        );
        """
        )

        self.conn.commit()

    def close(self):
        self.cursor.close()
        self.conn.close()

# ----------------- 特征提取器 -----------------

# class FeatureExtractor:

#     def __init__(self):
#         model = models.resnet50(pretrained=True)
#         self.model = torch.nn.Sequential(*list(model.children())[:-1])
#         self.model.eval()
#         self.transform = transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#         ])

#     def extract(self, image_path):
#         img = Image.open(image_path).convert("RGB")
#         img_tensor = self.transform(img).unsqueeze(0)
#         with torch.no_grad():
#             features = self.model(img_tensor).squeeze().numpy()
#         return features / np.linalg.norm(features)



# -------------------------------
# 1. 定义 SimCLR 模型结构（必须和训练时一致）
# -------------------------------

checkpoint_path = "/home/zhangbi/Zhangbi_Traveler/Figure_to_text/figure_cl/simclr_checkpoints/simclr_zhangbi_final.pth"
class SimCLR(nn.Module):
    def __init__(self, base_model=models.resnet50(pretrained=False), projection_dim=128):
        super(SimCLR, self).__init__()
        self.encoder = base_model
        in_features = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity() 
        self.projection = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.ReLU(),
            nn.Linear(in_features, projection_dim)
        )
    
    def forward(self, x):
        h = self.encoder(x)  
        z = self.projection(h)  
        return F.normalize(z, dim=1)

# -------------------------------
# 2. 特征提取器类（使用 SimCLR 模型）
# -------------------------------
class FeatureExtractor:
    def __init__(self, checkpoint_path):
        self.model = SimCLR()
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')  # 兼容 CPU/GPU
        state_dict = checkpoint['model_state_dict']
        
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[7:]  # 去掉 'module.' 前缀
            new_state_dict[k] = v
        self.model.load_state_dict(new_state_dict)
        
        self.model.eval()

        self.encoder = self.model.encoder  # 这就是 ResNet50 去掉 fc 后的 backbone

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extract(self, image_path):
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise ValueError(f"无法加载图像 {image_path}: {e}")

        img_tensor = self.transform(img).unsqueeze(0)  # 添加 batch 维度

        with torch.no_grad():
            features = self.encoder(img_tensor)  # shape: [1, 2048]
            features = features.squeeze().numpy()  # 转为 numpy 数组

        features = features
        return features


# ----------------- 图像特征管理器 -----------------

class ImageFeatureManager:

    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.extractor = FeatureExtractor(checkpoint_path)

    def clear_all_data(self):
        """Clears all data from the database (both spots and images)."""
        self.db_manager.cursor.execute("DELETE FROM images")
        self.db_manager.cursor.execute("DELETE FROM spots")
        self.db_manager.conn.commit()

    def add_image(self, spot_name, image_path):
        if self._image_exists(image_path):
            return
        self._add_image_to_db(spot_name, image_path)

    def _image_exists(self, image_path):
        self.db_manager.cursor.execute(
            "SELECT COUNT(*) FROM images WHERE image_path = %s", (image_path,)
        )
        return self.db_manager.cursor.fetchone()[0] > 0

    def _add_image_to_db(self, spot_name, image_path):
        self.db_manager.cursor.execute("SELECT id FROM spots WHERE name = %s", (spot_name,))
        result = self.db_manager.cursor.fetchone()

        if result is None:
            self.db_manager.cursor.execute("INSERT INTO spots (name) VALUES (%s)", (spot_name,))
            self.db_manager.conn.commit()
            spot_id = self.db_manager.cursor.lastrowid
        else:
            spot_id = result[0]

        features = self.extractor.extract(image_path)
        self.db_manager.cursor.execute(
            "INSERT INTO images (spot_id, image_path, features) VALUES (%s, %s, %s)",
            (spot_id, image_path, features.tobytes())
        )
        self.db_manager.conn.commit()
        print(f"Added image {image_path} to {spot_name}")

    def remove_image(self, image_path):
        self.db_manager.cursor.execute("DELETE FROM images WHERE image_path = %s", (image_path,))
        self.db_manager.conn.commit()

    
    def remove_spot(self, spot_name):
        self.db_manager.cursor.execute("""
            DELETE i FROM images i 
            JOIN spots s ON i.spot_id = s.id 
            WHERE s.name = %s
        """, (spot_name,))
        
        self.db_manager.cursor.execute("DELETE FROM spots WHERE name = %s", (spot_name,))
        self.db_manager.conn.commit()

    def find_similar(self, input_image, top_k=5):
        input_features = self.extractor.extract(input_image)

        self.db_manager.cursor.execute("SELECT spot_id, image_path, features FROM images")
        results = self.db_manager.cursor.fetchall()

        similarities = []
        for spot_id, image_path, feature_blob in results:
            db_features = np.frombuffer(feature_blob, dtype=np.float32)
            similarity = np.dot(input_features, db_features) / (
                np.linalg.norm(input_features) * np.linalg.norm(db_features)
            )

            spot_name = self.db_manager.cursor.execute("SELECT name FROM spots WHERE id = %s", (spot_id,))
            spot_name = self.db_manager.cursor.fetchone()[0]

            similarities.append((spot_name, image_path, float(similarity)))

        similarities.sort(key=lambda x: x[2], reverse=True)
        
        return similarities[:top_k]

    def get_images_by_spot(self, spot_name):
        
        self.db_manager.cursor.execute("""
            SELECT i.image_path 
            FROM images i 
            JOIN spots s ON i.spot_id = s.id 
            WHERE s.name = %s
        """, (spot_name,))
        
        results = self.db_manager.cursor.fetchall()
        image_paths = [row[0].split("/")[-1] for row in results]
    
        return image_paths

    def add_images_from_json(self, json_file):

        with open(json_file, 'r') as file:
            data = json.load(file)

        import tqdm
        for entry in tqdm.tqdm(data):
            spot_name = entry.get('metadata', {}).get('name')
            image_paths = entry.get('jpg_path', [])

            if not spot_name:
                continue
            
            for jpg_path in image_paths:
                if os.path.exists(jpg_path):
                    # print(jpg_path)
                    self.add_image(spot_name, jpg_path)
                else:
                    pass

    def get_all_spots(self):
        """Returns a list of all spot names from the database."""
        self.db_manager.cursor.execute("SELECT name FROM spots")
        results = self.db_manager.cursor.fetchall()
        spot_names = [row[0] for row in results]
    
        return spot_names
    
    def add_spot(self, spot_name):
        """
        Adds a new spot to the database if it does not already exist.
        
        :param spot_name: The name of the spot to be added.
        :return: True if the spot was added successfully or False if the spot already exists.
        """
        # Check if the spot already exists in the database
        self.db_manager.cursor.execute("SELECT COUNT(*) FROM spots WHERE name = %s", (spot_name,))
        count = self.db_manager.cursor.fetchone()[0]
        
        if count > 0:
            # Spot already exists, no need to add again
            return False
        
        # Insert the new spot into the database
        self.db_manager.cursor.execute("INSERT INTO spots (name) VALUES (%s)", (spot_name,))
        self.db_manager.conn.commit()
        
        return True

# ----------------- RAG 检索增强生成类 -----------------

class ZhangbiRAG:
    def __init__(self, json_path, vectorstore_path, model_path, device="cpu"):
        self.json_path = json_path
        self.vectorstore = self._initialize_vectorstore(vectorstore_path, model_path, device)
    
    def _initialize_vectorstore(self, vectorstore_path, model_path, device):
        embedding = HuggingFaceBgeEmbeddings(
            model_name=model_path,
            model_kwargs={"device": device}
        )
        return Chroma(
            persist_directory=vectorstore_path,
            embedding_function=embedding,
            collection_metadata={"hnsw:space": "cosine"},
        )

    def query(self, query, top_k=20):
        try:
            docs = self._retrieve_documents(query, top_k)
            return self._format_docs(docs)
        except Exception as e:
            return "查询失败，请检查日志。"

    def _retrieve_documents(self, query, top_k):
        doc11 = []

        with open(self.json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        for entry in data:
            if any(keyword in query for keyword in entry.get("contents", [])):
                try:
                    docs = self.vectorstore.similarity_search(
                        query, 
                        filter={"name": entry.get("metadata", {}).get("name")}, 
                        k=top_k
                    )
                    doc11.extend(docs)
                except Exception as e:
                    pass
        if not doc11:
            doc11 = self.vectorstore.similarity_search(query, k=top_k)
        
        return doc11

    def _format_docs(self, docs):
        if not docs:
            return "没有相关资料请补充。"
        return "\n".join(list(set([d.page_content for d in docs])))


# ----------------- 描述生成器类 -----------------

class DescriptionGenerator:
    def __init__(self, api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, spot_name, reference_text, model="qwen1.5-110b-chat"):
        messages = [
            {
                "role": "assistant",
                "content": "你是一个高效的信息提炼与中文内容撰写专家，代号为“奎木狼”。你的风格专业、生动，适用于文化旅游领域的内容创作。"
            },
            {
                "role": "user",
                "content": f"""
                    你的任务是基于下方提供的参考资料，围绕“{spot_name}”这一景点或主题，提炼出关键背景、特色亮点、历史文化等信息，并撰写一段简洁、生动、具有吸引力的中文介绍。请确保语言通顺自然，逻辑清晰，有一定文学性。

                    ✦ 写作要求：
                    - 内容应为一段完整的中文描述；
                    - 用词尽量丰富，避免直接复制参考内容的句式或表述；
                    - 如有典故、人物、特色事件，请自然融入文字；
                    - 切忌堆砌信息或机械列举；
                    - 若未找到关于“{spot_name}”的有效信息，请直接返回：“没有相关资料，请补充更多信息。”

                    ▍参考资料：
                    {reference_text}

                    ▍目标主题：
                    {spot_name}
            """
            }
        ]


        try:
            completion = self.client.chat.completions.create(
                model="qwen1.5-110b-chat",
                messages=messages,
                stream=False 
            )
            response = completion.choices[0].message.content
            return response.strip()
        except Exception as e:
            return "描述生成失败，请检查日志。"


# ----------------- 文本描述管理器 -----------------

class TextDescriptionManager:
    def __init__(self, db_manager, rag: ZhangbiRAG, generator: DescriptionGenerator):
        self.db_manager = db_manager
        self.rag = rag
        self.generator = generator
        self.ensure_description_column()

    def ensure_description_column(self):
        self.db_manager.cursor.execute("SHOW COLUMNS FROM spots LIKE 'description'")
        result = self.db_manager.cursor.fetchone()
        if result is None:
            self.db_manager.cursor.execute(
                "ALTER TABLE spots ADD COLUMN description TEXT DEFAULT NULL"
            )
            self.db_manager.conn.commit()
        else:
            pass

    def generate_text(self, spot_name):
        reference_text = self.rag.query(spot_name)
        return self.generator.generate(spot_name, reference_text)

    def get_text(self, spot_name):

        self.db_manager.cursor.execute(
            "SELECT description FROM spots WHERE name = %s", (spot_name,)
        )
        result = self.db_manager.cursor.fetchone()
        
        return result
        

    def add_text(self, spot_name):

        self.db_manager.cursor.execute(
            "SELECT id FROM spots WHERE name = %s", (spot_name,)
        )
        result = self.db_manager.cursor.fetchone()

        spot_id = None

        if result is None:

            self.db_manager.cursor.execute(
                "INSERT INTO spots (name) VALUES (%s)", (spot_name,)
            )
            self.db_manager.conn.commit()
            spot_id = self.db_manager.cursor.lastrowid  
        else:
            spot_id = result[0]
   
        self.db_manager.cursor.execute(
            "SELECT description FROM spots WHERE id = %s", (spot_id,)
        )
        description_result = self.db_manager.cursor.fetchone()

        if description_result is None or description_result[0] is None:

            generated_text = self.generate_text(spot_name)
            if generated_text and generated_text.strip() != "":
                self.db_manager.cursor.execute(
                    "UPDATE spots SET description = %s WHERE id = %s",
                    (generated_text, spot_id)
                )
                self.db_manager.conn.commit()
        else:
            pass

    def update_text(self, spot_name, new_text):
        """Updates the text description for the specified spot."""
        self.db_manager.cursor.execute(
            "SELECT id FROM spots WHERE name = %s", (spot_name,)
        )
        result = self.db_manager.cursor.fetchone()

        if result is None:
            return

        self.db_manager.cursor.execute(
            "UPDATE spots SET description = %s WHERE name = %s",
            (new_text, spot_name)
        )
        self.db_manager.conn.commit()


