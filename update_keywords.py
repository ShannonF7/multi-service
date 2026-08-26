import json
import os
import jieba

DATA_PATH = '/home/zhangbi/Zhangbi_Traveler/DataBase/Search_Update_Context/json/pgvector_optimized/src/intent/data/knowledge_units.json'

def generate_bigrams(text):
    # Strip non-alphanumeric roughly
    words = list(jieba.cut(text))
    # Filter stopwords slightly? For now just take all capable tokens
    words = [w for w in words if len(w.strip()) > 0]
    
    bigrams = []
    for i in range(len(words) - 1):
        bigrams.append(words[i] + words[i+1])
    return bigrams

try:
    with open(DATA_PATH, 'r') as f:
        data = json.load(f)
    
    updated_count = 0
    for item in data:
        if 'keywords' not in item or not item['keywords']: # Only update if missing
            text = item.get('text', '')
            bi = generate_bigrams(text)
            # Just take top 20 bigrams as "keywords" for simulation
            item['keywords'] = bi[:20]
            updated_count += 1
            
    with open(DATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        
    print(f"Successfully updated {updated_count} items with keywords.")
except Exception as e:
    print(f"Error: {e}")
