import json
import os
import sys

# Add the project root to sys.path to allow imports from src
current_dir = os.path.dirname(os.path.abspath(__file__))
# current_dir is .../json/pgvector_optimized/src/intent
# project_root is .../json/pgvector_optimized
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.llm.utils import call_api_with_retry

def load_data(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def generate_categories(data):
    if not data or 'items' not in data:
        print("Invalid data structure")
        return None

    items = data.get('items', [])
    items_for_llm = []
    
    # Iterate through all items and their content_blocks
    for item in items:
        # Get parent context (names)
        contents = item.get('contents', [])
        if isinstance(contents, str):
            parent_names = [contents]
        elif isinstance(contents, list):
            parent_names = contents
        else:
            parent_names = []
        parent_name_str = ", ".join([str(n) for n in parent_names])
        
        # Iterate blocks
        blocks = item.get('content_blocks', [])
        for block in blocks:
            block_id = block.get('id')
            block_content = block.get('content', '')
            
            # Use the actual ID from the json
            if block_id:
                # Truncate content for LLM context window
                if block_content and len(block_content) > 200:
                    display_content = block_content[:200] + "..."
                else:
                    display_content = block_content
                
                item_summary = f"ID: {block_id} | 归属: {parent_name_str} | 内容: {display_content}"
                items_for_llm.append(item_summary)

    spots_text = "\n".join(items_for_llm)
    
    prompt = f"""
你是一位构建“通用景区知识图谱”的专家。请基于参考的【景区文本段列表】，将其映射到一个**通用的、与具体景区解耦的领域知识树**中。

**核心目标**：
构建一个可以复用于任何古堡、古城或历史景区的通用知识分类体系。

**必须使用的顶层一级分类（KnowledgeDomain）**：
1. **建筑 (Architecture)**：涉及物理实体构造、空间布局、建筑构件等。
2. **军事 (Military)**：涉及防御体系、战争功能、战略设施、地道系统等。
3. **信仰 (Belief)**：涉及宗教场所、神灵崇拜、祭祀活动、民间信仰等。
4. **民俗 (Folklore)**：涉及生活方式、民间艺术、节庆习俗、非遗传承、饮食制作等。
5. **历史 (History)**：涉及年代背景、历史人物、演变过程、考古发现、碑刻记载等。
6. **服务 (Service)**：涉及旅游服务、设施指南、交通、食宿、安全提示等。

**任务要求**：
1. **构建通用概念子节点**：
    - 在上述一级分类下，设计二级/三级分类。
    - **关键**：子节点名称必须是**通用概念**（General Concepts）和**事实性描述**（Factual Description）。
    - ❌ 错误示例：“张壁地道”、“空王佛殿”。
    - ✅ 正确示例：“地下防御工事”、“佛教殿宇”、“防御型堡门”、“石窟造像”。
2. **归类（Mapping）**：
    - 将输入列表中的**每一个文本ID**，归入其最匹配的通用概念节点下。
    - **必须保证所有400多个ID都被归类**。
3. **输出格式**：
    - 严格的 **JSON** 格式。
    - 结构：
    {{
      "knowledge_tree": [
        {{
          "domain": "军事",
          "concepts": [
            {{
              "concept": "地下防御工事",
              "item_ids": ["zhangbigubao_00000", ...]
            }}, 
            ...
          ]
        }},
        ...
      ]
    }}

**输入数据（文本列表，共{len(items_for_llm)}条）**：
{spots_text}
"""
    
    print(f"正在构建通用领域知识树 (处理 {len(items_for_llm)} 条文本段)...")
    messages = [{"role": "user", "content": prompt}]
    response = call_api_with_retry(messages)
    
    # Create valid ID mapping for later verification
    original_items_map = {}
    for summary in items_for_llm:
        # Extract ID back: "ID: {block_id} | ..."
        parts = summary.split(' | ', 1)
        if parts:
            bid = parts[0].replace('ID: ', '').strip()
            original_items_map[bid] = summary
            
    return response, original_items_map


if __name__ == "__main__":
    # Path to the json file
    json_path = "/home/zhangbi/Zhangbi_Traveler/DataBase/Search_Update_Context/filled_json.json"
    
    data = load_data(json_path)
    if data:
        response_text, original_items_map = generate_categories(data)
        
        if response_text:
            cleaned_text = response_text.replace("```json", "").replace("```", "").strip()
            
            try:
                result = json.loads(cleaned_text)
                
                print("\n\n======== 通用景区知识图谱 (Generic Knowledge Domain) ========")
                
                tree = result.get('knowledge_tree', [])
                
                total_mapped_ids = 0
                all_ids_set = set(original_items_map.keys())
                mapped_ids_set = set()
                
                # 定义顺序
                current_domains = {node['domain']: node for node in tree}
                ordered_domains = ["建筑", "军事", "信仰", "民俗", "历史", "服务"]
                
                domains_to_print = [d for d in ordered_domains if d in current_domains]
                others = [d for d in current_domains if d not in ordered_domains]
                domains_to_print.extend(others)
                
                for domain_name in domains_to_print:
                    node = current_domains[domain_name]
                    print(f"\n[{domain_name}]")
                    for concept_node in node.get('concepts', []):
                        concept = concept_node.get('concept', 'General')
                        print(f"  ├── {concept}")
                        item_ids = concept_node.get('item_ids', [])
                        
                        count = 0
                        for i_id in item_ids:
                            # Convert to string just in case
                            i_id = str(i_id)
                            mapped_ids_set.add(i_id)
                            count += 1
                            # Optional: print first few to verify
                            if count <= 3:
                                raw_summary = original_items_map.get(i_id, "Unknown Entity")
                                # Summary: "ID: ... | 归属: ... | 内容: ..."
                                parts = raw_summary.split(' | ')
                                if len(parts) >= 2:
                                    # Print "归属: ..."
                                    print(f"  │    - [{i_id}] {parts[1]}")
                                else:
                                    print(f"  │    - [{i_id}]")
                        
                        if count > 3:
                             print(f"  │    ... (共 {count} 项)")
                        
                        total_mapped_ids += len(item_ids)

                print("\n" + "="*50)
                print(f"完整性检查:")
                print(f"原始文本总数: {len(all_ids_set)}")
                print(f"已归类ID总数: {len(mapped_ids_set)}")
                print(f"总映射次数: {total_mapped_ids} (允许同一ID归属多类)")
                
                missing_ids = all_ids_set - mapped_ids_set
                if missing_ids:
                    print(f"⚠️ 遗漏 {len(missing_ids)} 个ID")
                    # Save missing to file
                    with open("missing_ids.txt", "w") as f:
                        for mid in missing_ids:
                            f.write(f"{mid}: {original_items_map.get(mid)}\n")
                    print("遗漏ID已保存至 missing_ids.txt")
                
                # Save
                output_path = os.path.join(os.path.dirname(json_path), "knowledge_domain_tree_full.json")
                with open(output_path, "w", encoding="utf-8") as f:
                    final_data = {
                        "knowledge_tree": result,
                        "meta_map": original_items_map
                    }
                    json.dump(final_data, f, indent=2, ensure_ascii=False)
                print(f"\n完整知识图谱已保存至: {output_path}")

            except json.JSONDecodeError as e:
                print(f"JSON解析失败: {e}")
                print("原始返回内容:")
                print(cleaned_text)


            except json.JSONDecodeError as e:
                print(f"JSON解析失败: {e}")
                print("原始返回内容:")
                print(cleaned_text)
        else:
            print("未能生成分类目录。")
