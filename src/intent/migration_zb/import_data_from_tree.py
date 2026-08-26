import json
import os
import uuid

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Assumes script is in src/intent/
# BASE_DIR = .../json/pgvector_optimized/src/intent
# We want .../Search_Update_Context
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(BASE_DIR))))
TREE_PATH = os.path.join(PROJECT_ROOT, 'knowledge_domain_tree_full.json')
FILLED_PATH = os.path.join(PROJECT_ROOT, 'filled_json.json')
DEST_DATA_DIR = os.path.join(BASE_DIR, 'data')

# Space Code to Name Mapping (Manual Enrichment based on observed prefixes)
# Prefixes observed: baoqiang, beibaomen, wengcheng, didao, nanbaomen, 
# laolaoyuan, qingningtang, guandimiao, erlangmiao, kehanmiao, kuixinglou,
# longhefu, lvzuge, huaibaoliu, jindaigumu, kongwangxingci ...
CODE_TO_SPACE_INFO = {
    "baoqiang": {"name": "堡墙", "id": "ZBGB_WALL"},
    "beibaomen": {"name": "北堡门", "id": "ZBGB_NORTH_GATE"},
    "nanbaomen": {"name": "南堡门", "id": "ZBGB_SOUTH_GATE"},
    "wengcheng": {"name": "瓮城", "id": "ZBGB_WENGCHENG"},
    "didao": {"name": "地道", "id": "ZBGB_TUNNEL"},
    "laolaoyuan": {"name": "姥姥院", "id": "ZBGB_LAOLAOYUAN"},
    "nonggengyuan": {"name": "农耕院", "id": "ZBGB_NONGGENGYUAN"},
    "qingningtang": {"name": "清宁堂", "id": "ZBGB_QINGNINGTANG"},
    "guandimiao": {"name": "关帝庙", "id": "ZBGB_GUANDIMIAO"},
    "erlangmiao": {"name": "二郎庙", "id": "ZBGB_ERLANGMIAO"},
    "kehanmiao": {"name": "可汗庙", "id": "ZBGB_KEHANMIAO"},
    "kuixinglou": {"name": "魁星楼", "id": "ZBGB_KUIXINGLOU"},
    "longhefu": {"name": "龙鹤福", "id": "ZBGB_LONGHEFU"},
    "lvzuge": {"name": "吕祖阁", "id": "ZBGB_LVZUGE"},
    "huaibaoliu": {"name": "槐抱柳", "id": "ZBGB_HUAIBAOLIU"},
    "jindaigumu": {"name": "金代古墓", "id": "ZBGB_TOMB"},
    "kongwangxingci": {"name": "空王行祠", "id": "ZBGB_KONGWANGXINGCI"},
    "guzhaobi": {"name": "古照壁", "id": "ZBGB_ZHAOBI"},
    "zhangbigubao": {"name": "古堡整体", "id": "ZBGB_OVERALL"},
    "xinglongsi": {"name": "兴隆寺", "id": "ZBGB_XINGLONGSI"},
    "xifangshengjingdian": {"name": "西方圣境殿", "id": "ZBGB_XIFANGSHENGJING"},
    "zhenwumiao": {"name": "真武庙", "id": "ZBGB_ZHENWUMIAO"},
    "sandashidian": {"name": "三大士殿", "id": "ZBGB_SANDASHIDIAN"},
    "guhuaikezhan": {"name": "古槐客栈", "id": "ZBGB_GUHUAIKEZHAN"},
    "doufufang": {"name": "豆腐坊", "id": "ZBGB_DOUFUFANG"}
}

def get_space_info(prefix):
    # Try direct match
    if prefix in CODE_TO_SPACE_INFO:
        return CODE_TO_SPACE_INFO[prefix]
    
    # Fallback
    return {
        "name": f"未命名点位({prefix})",
        "id": f"ZBGB_{prefix.upper()}"
    }

def main():
    print(f"Reading Content from {FILLED_PATH}...")
    content_map = {} # id -> {text, assets_path}
    
    with open(FILLED_PATH, 'r', encoding='utf-8') as f:
        filled_data = json.load(f)
        
    for item in filled_data.get("items", []):
        metadata_file = item.get("metadata", {}).get("file", "")
        # Extract prefix from metadata file path if possible, e.g. .../didao
        file_prefix = os.path.basename(metadata_file)
        
        jpg_path = item.get("jpg_path", "")
        
        blocks = item.get("content_blocks", [])
        for block in blocks:
            b_id = block.get("id")
            b_text = block.get("content")
            
            if b_id and b_text:
                # Infer prefix from ID usually: didao_00001 -> didao
                prefix = b_id.split('_')[0] if '_' in b_id else file_prefix
                
                content_map[b_id] = {
                    "text": b_text,
                    "prefix": prefix,
                    "assets": [jpg_path] if jpg_path else [],
                    "topic": "其他" # Default
                }

    print(f"Loaded {len(content_map)} content blocks.")

    print(f"Reading Classification from {TREE_PATH}...")
    with open(TREE_PATH, 'r', encoding='utf-8') as f:
        tree_data = json.load(f)
    
    # The JSON structure seems to be: { "knowledge_tree": { "knowledge_tree": [ ...domains... ] } }
    # Based on the user provided snippet
    root = tree_data.get("knowledge_tree", {})
    if "knowledge_tree" in root:
        domains = root["knowledge_tree"]
    else:
        # Maybe the top level is the list
        domains = tree_data.get("knowledge_tree", [])
        
    # Apply Classification
    count_classified = 0
    for domain_obj in domains:
        topic_name = domain_obj.get("domain", "其他")
        concepts = domain_obj.get("concepts", [])
        
        for concept in concepts:
            item_ids = concept.get("item_ids", [])
            for iid in item_ids:
                if iid in content_map:
                    content_map[iid]["topic"] = topic_name
                    # Append Concept to text for better context? Optional.
                    # content_map[iid]["text"] = f"[{topic_name}-{concept['concept']}] " + content_map[iid]["text"]
                    count_classified += 1

    print(f"Applied topics to {count_classified} items.")
    
    # Generate Output Data
    final_spaces = {}
    final_kus = []
    
    for iid, data in content_map.items():
        prefix = data["prefix"]
        space_info = get_space_info(prefix)
        space_id = space_info["id"]
        
        # Add to Spaces List
        if space_id not in final_spaces:
            final_spaces[space_id] = {
                "space_id": space_id,
                "scenic_id": "ZBGB",
                "name": space_info["name"],
                "aliases": [] # Can be enriched later
            }
            
        # Add to KUs
        # Only add valid topics if we want to filter "其他"? 
        # For now keep all, but maybe default "其他" to "历史" or "民俗" based on simple Rules?
        # User said: "不相关的可以先不管", but better to include all.
        
        final_kus.append({
            "ku_id": iid.upper(), # Reuse the ID from filled_json
            "scenic_id": "ZBGB",
            "space_id": space_id,
            "topic": data["topic"], 
            "text": data["text"],
            "assets": data["assets"]
        })

    # Add "ZBGB_OVERALL" explicitly if missing
    if "ZBGB_OVERALL" not in final_spaces:
        final_spaces["ZBGB_OVERALL"] = {
            "space_id": "ZBGB_OVERALL",
            "scenic_id": "ZBGB",
            "name": "古堡整体",
            "aliases": ["古堡", "张壁"]
        }

    # Write Files
    spaces_list = list(final_spaces.values())
    
    with open(os.path.join(DEST_DATA_DIR, 'spaces.json'), 'w', encoding='utf-8') as f:
        json.dump(spaces_list, f, ensure_ascii=False, indent=2)
        
    with open(os.path.join(DEST_DATA_DIR, 'knowledge_units.json'), 'w', encoding='utf-8') as f:
        json.dump(final_kus, f, ensure_ascii=False, indent=2)

    print(f"Generated {len(spaces_list)} Spaces and {len(final_kus)} Knowledge Units.")

if __name__ == "__main__":
    main()
