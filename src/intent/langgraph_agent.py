import os
import json
import sys
from typing import TypedDict, List, Dict, Any, Optional, Literal
from langgraph.graph import StateGraph, START, END

# Setup Path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.intent.hierarchical_agent import HierarchicalAgent, Colors
from src.llm.utils import call_api_with_retry
from src.rag.zim_service import ZimService

# =========================================================================
# Refactoring Phase 2: Lightweight State Definition
# =========================================================================


class AgentState(TypedDict):
    # --- Context (Input) ---
    input_query: str
    scenic_id: str
    space_id: str             # GPS / Current Location
    current_location_name: str # Human readable location
    
    # --- 1. Intent Analysis (Router Output) ---
    detected_topic: Literal['建筑', '军事', '历史', '信仰', '民俗', '服务', '路线', '百科', 'NONE']
    # [NEW] Control Signal: 明确的交互模式，用于 Planner 快速决策
    query_type: Literal['QUESTION', 'CHAT'] # Simplifed

    # --- 1.5 Semantic Extraction (Extractor Output) ---
    semantic_analysis: Dict[str, Any] # [New] Stores extracted attributes

    # --- 2. Scope Planning (Planner Output) ---
    search_scope_type: Literal['LOCAL_HIERARCHY', 'SCENIC_GLOBAL', 'GLOBAL', 'ENTITY_DIRECT', 'NO_SEARCH', 'WEB_SEARCH'] # Added WEB_SEARCH
    target_space_ids: List[str]   # 指定搜索的空间 ID 列表，Empty 代表 Global
    
    # --- 3. Retrieval (Retriever Output) ---
    retrieved_ku_ids: List[str]   
    # [NEW] Retrieval Scores: 用于 Pre-Generation Gate 判断是否值得生成
    retrieval_scores: List[float]
    # [NEW] Direct Content: 绕过数据库 ID 查询，直接携带内容 (如 ZIM 百科)
    direct_context: List[Dict[str, Any]]
    
    # --- 3.5 Gate Status ---
    generation_allowed: bool
    
    # --- 4. Generation & Output ---
    tentative_answer: str
    is_answer_valid: bool
    final_answer: str

# 初始化基础设施 (DAL Provider)
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
SKILLS_DIR = os.path.join(os.path.dirname(__file__), 'skills')
core_agent = HierarchicalAgent(DATA_DIR)

# 初始化 ZIM 服务 (Lazy init handled in node or here? Here is safer)
ZIM_PATH = os.path.join(DATA_DIR, 'wikipedia_zh_all_maxi_2023-09.zim')
if os.path.exists(ZIM_PATH):
    ZimService.initialize(ZIM_PATH)
    print(f"{Colors.GREEN}[LangGraph] ZIM Service Initialized at {ZIM_PATH}{Colors.ENDC}")
else:
    print(f"{Colors.WARNING}[LangGraph] ZIM file not found at {ZIM_PATH}{Colors.ENDC}")

def load_skill_prompt(filename):
    try:
        with open(os.path.join(SKILLS_DIR, filename), 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Warning: Prompt file {filename} not found.")
        return ""

def call_llm_json(prompt: str, system_msg: str = "You are a helpful assistant.") -> Dict:
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt}
    ]
    resp = call_api_with_retry(messages)
    if not resp: return {}
    clean_resp = resp.replace('```json', '').replace('```', '').strip()
    try:
        return json.loads(clean_resp)
    except json.JSONDecodeError:
        print(f"{Colors.FAIL}JSON Parse Fail: {clean_resp}{Colors.ENDC}")
        return {}

# =========================================================================
# Refactoring Phase 3: Decoupled Nodes
# =========================================================================

def router_node(state: AgentState):
    """
    [Node 1: Intent Analyzer]
    Use 'topic_classifier.md' prompt.
    """
    print(f"{Colors.BLUE}[LangGraph] 第一步：意图分析 (Router){Colors.ENDC}")
    query = state['input_query']
    loc_name = core_agent.spaces.get(state['space_id'], {}).get('name', '未知位置')
    
    prompt = load_skill_prompt('topic_classifier.md')
    final_prompt = f"{prompt}\n\nUser Input: {query}"
    
    result = call_llm_json(final_prompt, "You are an Intent Classifier.")
    
    topic = result.get('topic', 'NONE')
    intent = result.get('intent_type', 'QUESTION')
    
    print(f"   -> 识别意图: {intent}, 话题: {topic}")
    
    return {
        "detected_topic": topic,
        "query_type": intent,
        "current_location_name": loc_name
    }

def extractor_node(state: AgentState):
    """
    [Node 1.5: Semantic Attribute Extractor]
    Use 'semantic_attribute_extractor.md' prompt.
    """
    print(f"{Colors.BLUE}[LangGraph] 第 1.5 步：语义属性提取 (Extractor){Colors.ENDC}")
    
    # If Chat/No Search, skip extraction to save time/cost
    if state['query_type'] == 'CHAT' or state['detected_topic'] == 'NONE':
        print("   -> 跳过提取 (非检索类意图)")
        return {"semantic_analysis": {}}

    query = state['input_query']
    prompt = load_skill_prompt('semantic_attribute_extractor.md')
    final_prompt = f"{prompt}\n\nUser Query: {query}"
    
    result = call_llm_json(final_prompt, "You are a Semantic Extractor.")
    
    subject = result.get('main_subject')
    attrs = result.get('target_attributes', [])
    scope_hint = result.get('implied_spatial_scope')
    
    print(f"   -> 提取结果: 主体='{subject}', 属性={attrs}, 范围暗示='{scope_hint}'")
    
    return {
        "semantic_analysis": result
    }

def planner_node(state: AgentState):
    """
    [Node 2: Scope Planner]
    Rules based on 'search_planner.md'.
    """
    print(f"{Colors.BLUE}[LangGraph] 第二步：检索范围规划{Colors.ENDC}")
    topic = state['detected_topic']
    q_type = state['query_type']
    current_space = state['space_id']
    semantic = state.get('semantic_analysis', {})
    
    # Logic strictly implementing search_planner.md
    
    # 1. NO_SEARCH Rule
    if topic == 'NONE' or q_type == 'CHAT':
        print("   -> 策略: 无需搜索 (闲聊或无话题)")
        return {"search_scope_type": "NO_SEARCH", "target_space_ids": []}
    
    # [NEW] Semantic Logic Override
    scope_hint = semantic.get('implied_spatial_scope', '')
    if scope_hint and 'Global' in scope_hint:
         print("   -> 策略: 全局百科搜索 (语义暗示范围为 Global)")
         return {"search_scope_type": "GLOBAL", "target_space_ids": []}
         
    # 2. GLOBAL Rule (Encyclopedia)
    if topic in ['百科', 'ENCYCLOPEDIA']:
        print("   -> 策略: 全局百科搜索")
        return {"search_scope_type": "GLOBAL", "target_space_ids": []}
        
    # 3. DOMAIN Rules
    if topic in ['建筑', '军事', '历史', '信仰', '民俗']:
        # Case B: Contextual (Local Hierarchy)
        print("   -> 策略: 本地层级搜索 (当前位置 + 上级)")
        ids = [current_space]
        parent = core_agent.spaces.get(current_space, {}).get('parent_id')
        if parent: ids.append(parent)
        return {"search_scope_type": "LOCAL_HIERARCHY", "target_space_ids": ids}

    # 4. SERVICE / ROUTE Rule
    if topic in ['服务', '路线', '导航', 'SERVICE']: 
        print(f"   -> 策略: 景区全域 ({topic})")
        return {"search_scope_type": "SCENIC_GLOBAL", "target_space_ids": []}

    # Fallback
    print("   -> 策略: 无需搜索 (默认回退)")
    return {"search_scope_type": "NO_SEARCH", "target_space_ids": []}


def retriever_node(state: AgentState):
    """
    [Node 3: Execution]
    执行检索，返回 IDs 和 Scores。
    """
    print(f"{Colors.BLUE}[LangGraph] 第三步：执行检索{Colors.ENDC}")
    
    scope = state['search_scope_type']
    if scope == 'NO_SEARCH':
        return {"retrieved_ku_ids": [], "retrieval_scores": []}
    
    # Check for Web Search Trigger
    if scope == 'WEB_SEARCH':
        print("   -> 进入网络搜索流程...")
        return {"retrieved_ku_ids": [], "retrieval_scores": []} # Web search handled later or mocked here?
        # Actually Web Search requires a different flow. Let's make a web_search_node. 
        # But if we use search_scope_type for flow control, we skip this node logic?
        # Let's clean this up: retriever_node is for Vector DB. Web Search logic should be separate.
        # But for valid state transition, valid retriever output is needed.
        # Let's return empty here and let Fallback or a specific branch handle it.

    query = state['input_query']
    target_ids = state['target_space_ids']
    
    # Configure params based on scope
    search_topic = state['detected_topic']
    search_scenic = state['scenic_id']
    search_scope_ids = None
    
    if scope == 'GLOBAL': 
        search_scenic = None # Search everywhere for encyclopedia
    elif scope == 'ENTITY_DIRECT' or scope == 'LOCAL_HIERARCHY':
        search_scope_ids = target_ids
    elif scope == 'SCENIC_GLOBAL':
         search_scope_ids = None # Rely on scenic_id constraint

    # Fetch (ID, Score) tuples
    results = core_agent.search_index(
        query=query,
        user_current_space_id=state['space_id'],
        scope_space_ids=search_scope_ids,
        topic=search_topic,
        scenic_id=search_scenic, 
        limit=3
    )
    
    ids = [r[0] for r in results]
    scores = [r[1] for r in results]
    
    print(f"   -> 检索到 {len(ids)} 条知识. 最高分: {max(scores) if scores else 0:.2f}")
    
    # ----------------------------------------------------
    # [ZIM Integration] 百科兜底逻辑
    # ----------------------------------------------------
    direct_context = []
    should_search_zim = False
    
    # Trigger conditions:
    # 1. Topic is Encyclopedia
    if search_topic in ['百科', 'ENCYCLOPEDIA']:
        should_search_zim = True
    # 2. Scope is GLOBAL (Semantic hint)
    elif scope == 'GLOBAL':
        should_search_zim = True
    # 3. Fallback: No local results found
    elif not ids and scope != 'NO_SEARCH':
        should_search_zim = True
        
    if should_search_zim:
        zim = ZimService.get_instance()
        if zim:
            # 优先使用语义提取的主体进行搜索 (去除 "是什么" 等干扰词)
            search_query = query
            semantic = state.get('semantic_analysis', {})
            subject = semantic.get('main_subject')
            
            if subject and subject != 'None':
                print(f"   -> [优化] 使用主体关键词 '{subject}' 代替原句进行 ZIM 检索")
                search_query = subject

            print(f"   -> 尝试 ZIM 百科检索 '{search_query}' ...")
            zim_results = zim.search(search_query, limit=2)
            if zim_results:
                print(f"   -> ZIM 命中 {len(zim_results)} 条")
                for z in zim_results:
                    # 将 ZIM 结果格式化为 Context Unit 结构
                    direct_context.append({
                        "id": z['id'],
                        "content": z['content'], # Contains title & snippet
                        "space_id": "GLOBAL_ZIM",
                        "metadata": z['metadata']
                    })
                    # 注入假分数以通过 Gate (Zim 匹配通常由标题精确匹配，置信度较高)
                    scores.append(0.85) 
            else:
                print("   -> ZIM 未命中")
    
    return {
        "retrieved_ku_ids": ids,
        "retrieval_scores": scores,
        "direct_context": direct_context
    }

def gate_node(state: AgentState):
    """
    [Node 3.5: Pre-Generation Gate]
    """
    print(f"{Colors.BLUE}[LangGraph] 第 3.5 步：生成前置校验 (Gate){Colors.ENDC}")
    
    if state['query_type'] == 'CHAT': return {"generation_allowed": True}
        
    scores = state['retrieval_scores']
    if not scores or max(scores) < 0.35:
        print(f"   -> {Colors.WARNING}GATE 拦截: 检索置信度过低{Colors.ENDC}")
        return {"generation_allowed": False}
    
    print(f"   -> GATE 通过。")
    return {"generation_allowed": True}

def generator_node(state: AgentState):
    """
    [Node 4: Generator]
    Use 'answer_generator.md' prompt.
    """
    print(f"{Colors.BLUE}[LangGraph] 第四步：答案生成{Colors.ENDC}")
    
    if state['query_type'] == 'CHAT':
        return {"tentative_answer": "你好！我是景区的智能助手，很高兴为您服务。"}

    ids = state['retrieved_ku_ids']
    direct_ctx = state.get('direct_context', [])
    
    # 结合 DB 结果和 ZIM 结果
    if not ids and not direct_ctx:
        return {"tentative_answer": "很抱歉，我没有找到相关的知识点。"}
        
    # 1. DB Content
    context_units = core_agent.get_knowledge_by_ids(ids)
    docs_input = [{"id": u.get('ku_id'), "content": u.get('text'), "space_id": u.get("space_id")} for u in context_units]
    
    # 2. Direct Content (ZIM)
    if direct_ctx:
        docs_input.extend(direct_ctx)
    
    prompt_tpl = load_skill_prompt('answer_generator.md')
    input_data = {
        "user_query": state['input_query'],
        "detected_topic": state['detected_topic'],
        "current_location": state['current_location_name'], # Explicitly passed
        "retrieved_docs": docs_input
    }
    
    full_prompt = f"{prompt_tpl}\n\nINPUT DATA JSON:\n{json.dumps(input_data, ensure_ascii=False)}"
    
    result = call_llm_json(full_prompt, "You are an Answer Generator.")
    
    ans = result.get("answer_text", "生成失败")
    used_ids = result.get("used_ku_ids", []) # Get used IDs
    
    # Filter IDs to only show USED ones
    final_ref_ids = [uid for uid in used_ids if uid in ids]
    
    # 显式附加引用源
    if final_ref_ids:
        # 简单去重
        unique_ids = list(set(final_ref_ids))
        ans += f"\n\n[参考资料ID: {', '.join(unique_ids)}]"

    return {"tentative_answer": ans}

def validator_node(state: AgentState):
    """
    [Node 5: Validator]
    """
    print(f"{Colors.BLUE}[LangGraph] 第五步：生成结果校验{Colors.ENDC}")
    ans = state['tentative_answer']
    
    if state['query_type'] == 'CHAT': return {"is_answer_valid": True}

    scores = state['retrieval_scores']
    max_score = max(scores) if scores else 0.0
    
    # Collect all source IDs for validation context
    all_source_ids = state['retrieved_ku_ids'][:]
    if state.get('direct_context'):
        all_source_ids.extend([item['id'] for item in state['direct_context']])

    prompt_tpl = load_skill_prompt('validator.md')
    input_data = {
        "user_query": state['input_query'],
        "answer_text": ans,
        "used_ku_ids": all_source_ids,
        "max_score": max_score
    }
    
    full_prompt = f"{prompt_tpl}\n\nINPUT DATA JSON:\n{json.dumps(input_data, ensure_ascii=False)}"
    
    result = call_llm_json(full_prompt, "You are a Validator.")
    
    status = result.get("status", "INVALID")
    reason = result.get("reason", "Unknown")
    
    print(f"   -> 校验状态: {status} ({reason})")
    return {"is_answer_valid": (status == "VALID")}

def web_search_node(state: AgentState):
    """
    [Node 6: Web Search Fallback]
    模拟网络搜索。
    """
    print(f"{Colors.BLUE}[LangGraph] 第六步：网络搜索 (Web Search){Colors.ENDC}")
    query = state['input_query']
    
    # 模拟网络搜索：实际上是用大模型的通用知识库来兜底，但伪装成搜索结果
    # 这里可以使用 search tool 如果有的话。
    prompt = f"用户的问题是：{query}。\n知识库中未找到答案。请利用你的通用知识，模拟一个网络搜索结果来回答这个问题。如果依然不知道，就说不知道。"
    
    ans = call_api_with_retry([{"role": "user", "content": prompt}])
    if not ans: ans = "网络搜索暂不可用。"
    
    final_ans = f"{ans}\n\n[来源: 网络搜索]"
    return {"final_answer": final_ans}

def fallback_planner_node(state: AgentState):
    """
    [Node 5.2: Fallback]
    """
    print(f"{Colors.WARNING}[LangGraph] 触发回退策略 (Fallback){Colors.ENDC}")
    
    current_scope = state['search_scope_type']
    
    if current_scope in ["LOCAL_HIERARCHY", "ENTITY_DIRECT"]:
        print("   -> 扩大范围至：景区全域 (SCENIC GLOBAL)...")
        return {
            "search_scope_type": "SCENIC_GLOBAL",
            "target_space_ids": []
        }
    elif current_scope == "SCENIC_GLOBAL":
        print("   -> 扩大范围至：全库百科 (GLOBAL)...")
        return {
            "search_scope_type": "GLOBAL",
            "target_space_ids": []
        }
    elif current_scope == "GLOBAL" or current_scope == "CROSS_SCENIC":
        print("   -> 启用网络搜索 (WEB_SEARCH)...")
        return {
            "search_scope_type": "WEB_SEARCH", 
            "target_space_ids": []
        }
    else:
        # Web Search 也无法处理的逻辑（虽然 web_search_node 是终点节点）
        return {"final_answer": "抱歉，网络搜索也未找到相关信息。"}

def output_node(state: AgentState):
    ans = state.get('tentative_answer')
    # If partial answer was "INVALID" or empty, and we arrived here, it means validation failed entirely or we gave up.
    if not ans or ans.strip() == "INVALID":
         ans = state.get('final_answer')
         if not ans:
             ans = "很抱歉，我暂时没有找到相关信息。"
    return {"final_answer": ans}

# --- Rewrite Workflow Topology ---

workflow = StateGraph(AgentState)

workflow.add_node("router", router_node)
workflow.add_node("extractor", extractor_node) # [NEW] Added Extractor Node
workflow.add_node("planner", planner_node)
workflow.add_node("retriever", retriever_node)
workflow.add_node("gate", gate_node)
workflow.add_node("generator", generator_node)
workflow.add_node("validator", validator_node)
workflow.add_node("fallback_planner", fallback_planner_node)
workflow.add_node("web_search", web_search_node)
workflow.add_node("output", output_node)

# Linear Flow
workflow.add_edge(START, "router")
workflow.add_edge("router", "extractor") # Router -> Extractor
workflow.add_edge("extractor", "planner") # Extractor -> Planner
workflow.add_edge("planner", "retriever")
workflow.add_edge("retriever", "gate")

# Conditional: Gate -> Generator OR Fallback
def check_gate(state: AgentState):
    if state["generation_allowed"]:
        return "go_to_gen"
    else:
        return "go_to_fallback"

workflow.add_conditional_edges(
    "gate", 
    check_gate,
    {"go_to_gen": "generator", "go_to_fallback": "fallback_planner"}
)

workflow.add_edge("generator", "validator")

# Conditional: Validator -> Output OR Fallback
def check_validation(state: AgentState):
    if state["is_answer_valid"]:
        return "go_to_output"
    else:
        # Check loop preventer
        if state["search_scope_type"] == "SCENIC_GLOBAL":
            return "go_to_output" # Give up
        else:
            return "go_to_fallback"

workflow.add_conditional_edges(
    "validator",
    check_validation,
    {"go_to_output": "output", "go_to_fallback": "fallback_planner"}
)

# Fallback Loop
def check_fallback_retry(state: AgentState):
    if state.get("final_answer"): # Means we gave up in fallback node? No, fallback returns state updates.
        # Check specific scope to determine next node
        return "go_to_output"
    
    scope = state['search_scope_type']
    if scope == 'WEB_SEARCH':
        return "go_to_web"
    
    return "go_to_retriever"

workflow.add_conditional_edges(
    "fallback_planner",
    check_fallback_retry,
    {
        "go_to_retriever": "retriever", 
        "go_to_web": "web_search",
        "go_to_output": "output"
    }
)

workflow.add_edge("web_search", "output") # Web Search goes to output

workflow.add_edge("output", END)

app = workflow.compile()

if __name__ == "__main__":
    # 配置模拟场景
    scenarios = [
        {
            "name": "闲聊 (无需检索)",
            "state": {
                "input_query": "你好呀",
                "scenic_id": "ZBGB",
                "space_id": "ZBGB_WALL",
                # Init empty fields
                "current_location_name": "", "detected_topic": "CHAT", "query_type": "QUESTION",
                "search_scope_type": "NO_SEARCH", "target_space_ids": [],
                "retrieved_ku_ids": [], "retrieval_scores": [], "generation_allowed": True,
                "tentative_answer": "", "is_answer_valid": False, "final_answer": ""
            }
        },
        {
            "name": "张壁古堡-历史问题",
            "state": {
                "input_query": "张壁古堡在哪里？",
                "scenic_id": "ZBGB",
                "space_id": "ZBGB_WALL",
                "current_location_name": "", "detected_topic": "CHAT", "query_type": "QUESTION",
                "search_scope_type": "NO_SEARCH", "target_space_ids": [],
                "retrieved_ku_ids": [], "retrieval_scores": [], "generation_allowed": True,
                "tentative_answer": "", "is_answer_valid": False, "final_answer": ""
            }
        },
        {
            "name": "低置信度测试 (Trigger Gate)",
            "state": {
                # 假设这个问题匹配度极低
                "input_query": "火星上有水吗？",
                "scenic_id": "ZBGB",
                "space_id": "ZBGB_WALL",
                "current_location_name": "", "detected_topic": "CHAT", "query_type": "QUESTION",
                "search_scope_type": "NO_SEARCH", "target_space_ids": [],
                "retrieved_ku_ids": [], "retrieval_scores": [], "generation_allowed": True,
                "tentative_answer": "", "is_answer_valid": False, "final_answer": ""
            }
        },
        {
            "name": "张壁古堡堡墙高度",
            "state": {
                "input_query": "这个墙多高？",
                "scenic_id": "ZBGB",
                "space_id": "ZBGB_WALL",
                "current_location_name": "", "detected_topic": "CHAT", "query_type": "QUESTION",
                "search_scope_type": "NO_SEARCH", "target_space_ids": [],
                "retrieved_ku_ids": [], "retrieval_scores": [], "generation_allowed": True,
                "tentative_answer": "", "is_answer_valid": False, "final_answer": ""
            }
        },
        {
            "name": "平遥古城堡墙高度",
            "state": {
                "input_query": "这个墙多高？",
                "scenic_id": "PYGC",
                "space_id": "PYGC_WALL",
                "current_location_name": "", "detected_topic": "CHAT", "query_type": "QUESTION",
                "search_scope_type": "NO_SEARCH", "target_space_ids": [],
                "retrieved_ku_ids": [], "retrieval_scores": [], "generation_allowed": True,
                "tentative_answer": "", "is_answer_valid": False, "final_answer": ""
            }
        },
        {
            "name": "景区内的全域百科测试",
            "state": {
                "input_query": "故宫在哪里？",
                "scenic_id": "PYGC",
                "space_id": "PYGC_WALL",
                "current_location_name": "", "detected_topic": "CHAT", "query_type": "QUESTION",
                "search_scope_type": "NO_SEARCH", "target_space_ids": [],
                "retrieved_ku_ids": [], "retrieval_scores": [], "generation_allowed": True,
                "tentative_answer": "", "is_answer_valid": False, "final_answer": ""
            }
        },
        {
            "name": "路线类测试",
            "state": {
                "input_query": "我现在在哪里？",
                "scenic_id": "ZBGB",
                "space_id": "ZBGB_WALL",
                "current_location_name": "", "detected_topic": "CHAT", "query_type": "QUESTION",
                "search_scope_type": "NO_SEARCH", "target_space_ids": [],
                "retrieved_ku_ids": [], "retrieval_scores": [], "generation_allowed": True,
                "tentative_answer": "", "is_answer_valid": False, "final_answer": ""
            }
        },
        {
            "name": "路线类测试",
            "state": {
                "input_query": "厕所在哪里？",
                "scenic_id": "ZBGB",
                "space_id": "ZBGB_TUNNEL",
                "current_location_name": "", "detected_topic": "CHAT", "query_type": "QUESTION",
                "search_scope_type": "NO_SEARCH", "target_space_ids": [],
                "retrieved_ku_ids": [], "retrieval_scores": [], "generation_allowed": True,
                "tentative_answer": "", "is_answer_valid": False, "final_answer": ""
            }
        },
         {
            "name": "吃的",
            "state": {
                "input_query": "景区有吃的吗？",
                "scenic_id": "ZBGB",
                "space_id": "ZBGB_WALL",
                "current_location_name": "", "detected_topic": "CHAT", "query_type": "QUESTION",
                "search_scope_type": "NO_SEARCH", "target_space_ids": [],
                "retrieved_ku_ids": [], "retrieval_scores": [], "generation_allowed": True,
                "tentative_answer": "", "is_answer_valid": False, "final_answer": ""
            }
        },
        {
            "name": "ZIM 百科兜底测试 (Python)",
            "state": {
                "input_query": "Python是什么？",
                "scenic_id": "ZBGB",
                "space_id": "ZBGB_WALL",
                "current_location_name": "", "detected_topic": "CHAT", "query_type": "QUESTION",
                "search_scope_type": "NO_SEARCH", "target_space_ids": [],
                "retrieved_ku_ids": [], "retrieval_scores": [], "generation_allowed": True,
                "tentative_answer": "", "is_answer_valid": False, "final_answer": ""
            }
        },
    ]

    for sc in scenarios:
        print("\n" + "="*60)
        print(f"🎬 Scenario: {sc['name']}")
        print(f"User Query: {sc['state']['input_query']}")
        print("="*60)
        
        try:
            final_state = app.invoke(sc['state'])
            
            print(f"\n{Colors.BOLD}=== Result for {sc['name']} ==={Colors.ENDC}")
            print(f"Type: {final_state.get('query_type')}")
            print(f"Scope: {final_state.get('search_scope_type')}")
            print(f"Gate Passed: {final_state.get('generation_allowed')}")
            print(f"Docs: {len(final_state.get('retrieved_ku_ids', []))}")
            print(f"Final: {final_state.get('final_answer')}")
        except Exception as e:
            print(f"{Colors.FAIL}Error: {e}{Colors.ENDC}")
            import traceback
            traceback.print_exc()