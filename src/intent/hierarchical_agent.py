import json
import os
import sys

# Setup Path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

import requests
import numpy as np
import logging

# [NEW] Import FlagEmbedding directly since we are mocking pgvector logic in-process
try:
    from FlagEmbedding import BGEM3FlagModel
    HAS_BGE = True
except ImportError:
    HAS_BGE = False
    print("Warning: FlagEmbedding not found. Vector search will be simulated with random scores.")

try:
    from src.llm.utils import call_api_with_retry
except ImportError:
    # 仅作为最后的回退，避免程序崩溃
    def call_api_with_retry(messages, **kwargs):
        return "LLM 接口不可用，请检查环境配置。"

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

class RAGService:
    def __init__(self, endpoint="http://localhost:7002/smart_query/"):
        self.endpoint = endpoint

    def query(self, user_query, space_id=None):
        """
        Call the external RAG service to get relevant Vector Search results.
        Support space_id filtering if available.
        """
        params = {"query": user_query}
        if space_id:
            params["space_id"] = space_id
            
        print(f"{Colors.BLUE}[RAG] Requesting: {user_query} (Space: {space_id}){Colors.ENDC}")
        try:
            response = requests.get(self.endpoint, params=params, timeout=3.5)
            if response.status_code == 200:
                data = response.json()
                # 统一转换格式
                if isinstance(data, dict) and data.get("status") == "success":
                    return [{"text": it.get("content"), "space_id": it.get("metadata", {}).get("space_id")} for it in data.get("results", [])]
            return []
        except Exception as e:
            print(f"{Colors.WARNING}[RAG] Error: {e}{Colors.ENDC}")
            return []

class TopicSkill:
    def __init__(self, prompt_path, valid_topics: list):
        with open(prompt_path, 'r', encoding='utf-8') as f:
            self.template = f.read()
        self.valid_topics = valid_topics

    def classify(self, query, local_context=""):
        topics_str = ", ".join(self.valid_topics)
        prompt = self.template.replace("{{user_query}}", query) \
                              .replace("{{valid_topics}}", topics_str) \
                              .replace("{{local_context}}", local_context)
        
        # 真正调用 LLM 进行分类
        resp = call_api_with_retry([{"role": "system", "content": "You are a classifier."}, {"role": "user", "content": prompt}])
        result = resp.strip().replace('"', '')
        
        # 纠偏逻辑：确保返回的是 topics.json 中的项
        for t in self.valid_topics:
            if t in result: return t
        return "百科"

class HierarchicalAgent:
    def __init__(self, data_dir):
        # 1. 动态加载数据与分类
        with open(os.path.join(data_dir, 'knowledge_units.json'), 'r') as f: self.kus = json.load(f)
        with open(os.path.join(data_dir, 'topics.json'), 'r', encoding='utf-8') as f: 
            self.valid_topics = json.load(f)
        with open(os.path.join(data_dir, 'spaces.json'), 'r') as f:
            spaces_list = json.load(f)
            self.spaces = {s['space_id']: s for s in spaces_list}
        
        # Load Prompts (Mocking or real if files exist)
        self.role_prompts = {}
        
        # [NEW] Initialize Vector Model (Simulation)
        self.embed_model = None
        self.ku_embeddings = {} # {ku_id: vector}
        self._init_vector_engine()

    def _init_vector_engine(self):
        """
        Initialize BGE model and pre-compute embeddings for small Knowledge Base.
        In production, this is handled by pgvector.
        """
        if not HAS_BGE: return
        
        model_path = "/home/zhangbi/Zhangbi_Traveler/LLM_model/Model_api/checkpoints/bge-large-zh-v1.5/"
        print(f"{Colors.BLUE}[System] Loading BGE Model from {model_path}...{Colors.ENDC}")
        try:
            self.embed_model = BGEM3FlagModel(model_path, device="cuda", use_fp16=True)
            print(f"{Colors.GREEN}[System] BGE Model Loaded.{Colors.ENDC}")
            
            # Compute embeddings for all KUs (Simulated Indexing)
            # Strategy: Embed "text" + "keywords"
            print("[System] Indexing Knowledge Units...")
            texts = []
            ids = []
            for ku in self.kus:
                # Construct rich representation
                kw_str = " ".join(ku.get('keywords', []))
                content = f"{ku['text']} {kw_str}"
                texts.append(content)
                ids.append(ku['ku_id'])
            
            # Batch encode
            embeddings = self.embed_model.encode(texts, batch_size=12, max_length=512)['dense_vecs']
            
            for i, ku_id in enumerate(ids):
                self.ku_embeddings[ku_id] = embeddings[i]
                
            print(f"{Colors.GREEN}[System] Indexed {len(self.kus)} docs.{Colors.ENDC}")
            
        except Exception as e:
            print(f"{Colors.FAIL}[System] Vector Engine Init Failed: {e}{Colors.ENDC}")
            self.embed_model = None

    def score_unit(self, unit, query, current_space_id, intent_topic=None):
        text = unit.get('text', '')
        if not query: return 0
        
        # 1. 基础文本匹配分
        match_score = sum(1 for char in query if char in text) / len(query)
        
        # 2. 地理空间加成 (层次化权重)
        u_space = unit.get('space_id')
        if u_space == current_space_id:
            match_score *= 2.0
        else:
            # 2.1 检查是否为父级空间 (User is at child, KU is at parent)
            curr_s = self.spaces.get(current_space_id, {})
            is_parent = False
            while curr_s.get('parent_id'):
                if u_space == curr_s['parent_id']:
                    match_score *= 1.2
                    is_parent = True
                    break
                curr_s = self.spaces.get(curr_s['parent_id'], {})
            
            # 2.2 检查是否为子级空间 (User is at parent, KU is at child)
            if not is_parent:
                # 预先找到当前空间的所有子孙空间 ID
                all_descendants = []
                temp_queue = [current_space_id]
                while temp_queue:
                    parent = temp_queue.pop(0)
                    children = [sid for sid, s in self.spaces.items() if s.get('parent_id') == parent]
                    all_descendants.extend(children)
                    temp_queue.extend(children)
                
                if u_space in all_descendants:
                    match_score *= 1.8 # 子级空间非常相关
            
        # 3. 显式景点名称提及加成 (点名加倍)
        scenic_id = unit.get('scenic_id')
        for sid, sinfo in self.spaces.items():
            if sinfo.get('scenic_id') == scenic_id:
                sname = sinfo.get('name')
                if sname and sname in query:
                    if u_space == sid:
                        match_score *= 3.0 # 点名该景点，权重极高
                        break
        
        # 4. 话题一致性加成 (1.5x)
        if intent_topic and unit.get('topic') == intent_topic:
            if match_score > 0: 
                match_score *= 1.5
                
        return match_score

    def generate_response(self, query, context_units, topic=None):
        if not context_units: return "对不起，我暂时没有找到相关的信息。"
        
        # 动态选取技能
        skill_file = self.skill_map.get(topic, 'encyclopedia.md')
        if skill_file not in self.role_prompts:
            with open(os.path.join(self.skills_dir, skill_file), 'r', encoding='utf-8') as f:
                self.role_prompts[skill_file] = f.read()
        
        # 格式化上下文，确保 ID 醒目
        context_str = "\n".join([f"### [知识源 ID: {u.get('ku_id', '未知')}]\n{u['text']}" for u in context_units])
        
        # 强化“强相关性”与“证据驱动”的指令
        instruction = """
# 强制指令 (优先级最高):
1. **证据驱动 (Evidence-Based)**: 你的回答【必须且仅能】基于下方 # Context 提供的文本。回答中的每一个结论都必须能从 Context 中找到直接对应关系。
2. **强相关性 (Strong Relevance)**: 针对用户的问题，请从 Context 中提取最直接、最精准的片段进行回答。严禁使用 Context 之外的外部知识（即便是你认为正确的外部常识也不行）。
3. **精准溯源 (Traceability)**: 在陈述事实时，请在句子末尾或段落末尾使用 [ID] 形式标注来源。例如：“堡墙厚度为12米 [BAOQIANG_001]”。
4. **回答质量**: 拒绝模棱两可。如果 Context 中只有部分信息相关，就只回答那一部分。如果完全不相关，请直接回答“知识库中未找到直接相关的证据”。
5. **拒绝开场白**: 不要说“根据资料显示”或“作为专家”之类的话，直接开始回答问题。
6. **引用列表 (Reference List)**: 在答案的最后一行，请显式提供引用的知识点 ID。格式要求：\n参考知识：KID1, KID2...
"""
        
        # 重新编排 Prompt 结构：指令 -> 背景人设 -> 知识内容 -> 用户问题
        final_prompt = f"""
{instruction}

# 你的背景人设:
{self.role_prompts[skill_file]}

# Context (输入的本地知识库内容):
{context_str}

# 用户的问题 (User Query):
{query}

# 你的最终回答:
"""
        return call_api_with_retry([{"role": "user", "content": final_prompt}])

    # =========================================================================
    # Refactoring Phase 1: Data Access Layer (DAL) Methods
    # 遵循架构约束：将计算/检索逻辑与 State 分离，支持 ID-Based 操作
    # =========================================================================

    def get_knowledge_by_ids(self, ku_ids):
        """
        [DAL] 根据 ID 列表获取完整的知识对象 (Hydration)。
        迁移 pgvector 后，这里对应: SELECT * FROM knowledge_units WHERE id IN (...)
        """
        if not ku_ids: return []
        # 建立临时索引优化查找 (实际 DB 不需要)
        ku_map = {ku['ku_id']: ku for ku in self.kus}
        return [ku_map[kid] for kid in ku_ids if kid in ku_map]

    def search_index(self, query, user_current_space_id, scope_space_ids=None, topic=None, limit=5, scenic_id=None):
        """
        [DAL] 执行搜索并返回 (id, score) 元组列表。
        Updated to use Vector Search (BGE) + Scope Filtering.
        """
        # 1. Scope Filtering (Base Candidates Selection)
        candidates = []
        if scope_space_ids is not None:
            scope_set = set(scope_space_ids)
            candidates = [ku for ku in self.kus if ku.get('space_id') in scope_set]
        elif scenic_id:
            candidates = [ku for ku in self.kus if ku.get('scenic_id') == scenic_id]
        else:
            candidates = self.kus

        if not candidates:
            return []

        # 2. Vector Search
        scored_results = []
        if self.embed_model:
            # Embed Query
            q_vec = self.embed_model.encode([query])['dense_vecs'][0]
            norm_q = np.linalg.norm(q_vec)
            
            for ku in candidates:
                ku_id = ku['ku_id']
                doc_vec = self.ku_embeddings.get(ku_id)
                if doc_vec is None: continue
                
                # A. Base Vector Score (Cosine Similarity)
                norm_doc = np.linalg.norm(doc_vec)
                base_score = np.dot(q_vec, doc_vec) / (norm_q * norm_doc)
                
                final_score = base_score
                
                # B. Explicit Keyword Match Bonus (Simulating Keyword Search)
                # Check for query distinct terms in KU keywords
                # Why? Vector search can sometimes be "fuzzy", explicit keywords confirm intent.
                if ku.get('keywords'):
                    hit_count = 0
                    for kw in ku['keywords']:
                        if len(kw) > 1 and kw in query: # Simple substring match
                            hit_count += 1
                    
                    if hit_count > 0:
                        # Add a small additive bonus + multiplier to reward exact keyword hits
                        # Mimics app_copy.py's strategy of prioritizing keyword matches
                        final_score = final_score * 1.05 + 0.05
                
                # C. Spatial Context Weighting (Topic-Dependent)
                # [Optimization] Differentiated boosting based on Intent Topic
                # Safety Check: Only boost if there is at least minimal semantic relevance (e.g. > 0.3)
                # This prevents boosting completely irrelevant docs just because they are "nearby".
                if ku.get('space_id') == user_current_space_id and final_score > 0.3:
                    if topic in ['路线', '服务', 'SERVICE', 'ROUTE']:
                        # Navigation intents strongly imply "nearby" preference.
                        # [User Request] Weight increased heavily for Route/Service
                        # [Fix] Only boost if the KU is actually relevant to Service/Route. 
                        # Don't boost irrelevant topics (e.g. Military History) just because they are nearby.
                        ku_topic = ku.get('topic', '')
                        if ku_topic in ['路线', '服务', '设施', '餐饮', '其他', 'Route', 'Service', 'Facility', 'Others', 'Common']:
                            final_score *= 2.0
                        else:
                            # Mismatch topic (e.g. Service query, but History doc). 
                            # Only slight boost for location context.
                            final_score *= 1.1
                    elif topic in ['建筑', 'BUILDING']:
                        # Contextual QA ("How tall is this wall?") -> likely refers to immediate surroundings.
                        final_score *= 1.2
                    elif topic in ['百科', 'ENCYCLOPEDIA', '历史', 'HISTORY']:
                        # General knowledge queries are often location-agnostic.
                        final_score *= 1.05
                    else:
                        # Default assumption for other topics -> mild preference for here.
                        final_score *= 1.1

                # Threshold filtering
                if final_score > 0.38: # Slightly raised from 0.35 to reduce noise (recall is now 5)
                    scored_results.append((final_score, ku_id))
        else:
            # Fallback to Mock Scoring (if vector model failed/missing)
            print(f"{Colors.WARNING}[Search] Using legacy fallback scoring{Colors.ENDC}")
            for ku in candidates:
                # Simple keyword overlap
                match = sum(1 for c in query if c in ku['text']) / (len(query) + 1)
                if ku.get('space_id') == user_current_space_id: match *= 2.0
                if match > 0.1:
                    scored_results.append((match, ku['ku_id']))

        # 3. Ranking
        scored_results.sort(key=lambda x: x[0], reverse=True)
        
        return [(kid, float(score)) for score, kid in scored_results[:limit]]

    def get_location_hierarchy(self, space_id):
        """
        [Helper] 获取空间层级链 (用于 Scope Planning)。
        返回: [current_id, parent_id, grant_parent_id, ...]
        """
        chain = []
        curr = space_id
        while curr:
            chain.append(curr)
            info = self.spaces.get(curr)
            if not info: break
            curr = info.get('parent_id')
        return chain

    def get_child_spaces(self, space_id):
        """[Helper] 获取直接子空间 ID"""
        return [s['space_id'] for s in self.spaces.values() if s.get('parent_id') == space_id]

    def search(self, scenic_id, current_gps_space_id, user_query):
        print(f"\n{Colors.HEADER}=== NEW QUERY: {user_query} ==={Colors.ENDC}")
        print(f"{Colors.BLUE}[Context] GPS: {current_gps_space_id} | Scenic: {scenic_id}{Colors.ENDC}")

        top_hits = []

        # 1. Try RAG Service First (Vector Search)
        rag_hits = self.rag_service.query(user_query)
        if rag_hits:
            print(f"{Colors.GREEN}   -> [RAG] Retrieved {len(rag_hits)} results from Vector DB.{Colors.ENDC}")
            top_hits = rag_hits
        else:
            print(f"{Colors.WARNING}   -> [RAG] Service unavailable or no results. Fallback to Local Search.{Colors.ENDC}")
            
            # 2. Local Fallback Search (Keyword + Heuristic)
            
            # 2.1 Local Scoped Search
            candidates = [ku for ku in self.kus if ku['scenic_id'] == scenic_id]
            scored_results = []
            for ku in candidates:
                score = self.score_unit(ku, user_query, current_gps_space_id)
                if score > 2.0:
                    scored_results.append((score, ku))
            
            scored_results.sort(key=lambda x: x[0], reverse=True)
            top_hits = [x[1] for x in scored_results[:3]]

            # 2.2 Global Fallback Search
            if not top_hits:
                print(f"{Colors.WARNING}   -> No local info. Searching Global Knowledge...{Colors.ENDC}")
                global_candidates = [ku for ku in self.kus if ku['scenic_id'] != scenic_id]
                scored_global = []
                for ku in global_candidates:
                    score = self.score_unit(ku, user_query, "GLOBAL_COMMON") 
                    if score > 2.0:
                        scored_global.append((score, ku))
                scored_global.sort(key=lambda x: x[0], reverse=True)
                top_hits = [x[1] for x in scored_global[:3]]

        # 3. Generate & Display
        if top_hits:
            print(f"{Colors.GREEN}   -> Found {len(top_hits)} context pieces matching query.{Colors.ENDC}")
            
            # Print Hits Verification
            for idx, unit in enumerate(top_hits):
                topic = unit.get('topic', 'N/A')
                source = unit.get('source', 'LOCAL_JSON') # Mark source
                snippet = unit['text'][:60].replace('\n', ' ')
                print(f"      {idx+1}. [{source} | {topic}] {snippet}...")
            
            final_reply = self.generate_response(user_query, top_hits)
            print(f"{Colors.BOLD}Agent: {final_reply}{Colors.ENDC}")
        else:
            print(f"{Colors.FAIL}Agent: 抱歉，我的知识库里没有关于这个问题的记录。{Colors.ENDC}")

    def run(self, scenic_id, current_gps_space_id, user_query):
        self.search(scenic_id, current_gps_space_id, user_query)

if __name__ == "__main__":
    agent = HierarchicalAgent(os.path.join(os.path.dirname(__file__), 'data'))

    # # Case 1: Local Specific (History)
    # agent.run("ZBGB", "ZBGB_WALL", "这墙有多厚？")

    # # Case 2: Local Service (Guide)
    # agent.run("ZBGB", "ZBGB_TUNNEL", "哪里有厕所？")

    # # Case 3: External Knowledge (Encyclopedia)
    # agent.run("ZBGB", "ZBGB_TUNNEL", "故宫有多少个房间？")
    
    # # Case 4: Ambiguous/Global
    # agent.run("ZBGB", "ZBGB_TUNNEL", "南京博物院有哪些宝贝？")

    # Case 5: 
    agent.run("ZBGB", "ZBGB_TUNNEL", "故宫有多少个房间？")
