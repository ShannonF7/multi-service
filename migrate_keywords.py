import json
import os

SOURCE_FILE = "/home/zhangbi/Zhangbi_Traveler/DataBase/Search_Update_Context/json/base_structure.json"
TARGET_FILE = "/home/zhangbi/Zhangbi_Traveler/DataBase/Search_Update_Context/json/pgvector_optimized/src/intent/data/knowledge_units.json"

def main():
    print(f"Loading source: {SOURCE_FILE}")
    with open(SOURCE_FILE, 'r', encoding='utf-8') as f:
        source_json = json.load(f)

    # Dictionary to map upper(ID) -> keywords
    keyword_map = {}
    
    count_source = 0
    # Process nested structure
    # Structure seems to be: { "items": [ { "content_blocks": [ { "id": "...", "keywords": [...] } ] } ] }
    if "items" in source_json:
        for item in source_json["items"]:
            if "content_blocks" in item:
                for block in item["content_blocks"]:
                    if "id" in block and "keywords" in block:
                        # Use uppercase ID for matching Key
                        key_id = block["id"].strip().upper()
                        keyword_map[key_id] = block["keywords"]
                        count_source += 1
    
    print(f"Loaded {count_source} keyword entries from source.")

    print(f"Loading target: {TARGET_FILE}")
    with open(TARGET_FILE, 'r', encoding='utf-8') as f:
        target_units = json.load(f)

    updated_count = 0
    cleared_count = 0

    for unit in target_units:
        ku_id = unit.get("ku_id")
        if not ku_id:
            continue
            
        # Match ID (case-insensitive)
        lookup_id = ku_id.strip().upper()
        
        if lookup_id in keyword_map:
            unit["keywords"] = keyword_map[lookup_id]
            updated_count += 1
        else:
            # ID 不匹配的则空着
            unit["keywords"] = []
            cleared_count += 1

    print(f"Updating target file...")
    print(f"  Matched & Updated: {updated_count}")
    print(f"  Not Matched & Cleared: {cleared_count}")

    with open(TARGET_FILE, 'w', encoding='utf-8') as f:
        json.dump(target_units, f, ensure_ascii=False, indent=4)
    
    print("Done.")

if __name__ == "__main__":
    main()
