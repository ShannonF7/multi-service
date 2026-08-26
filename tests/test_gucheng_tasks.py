import requests
import json
import sys

# 配置信息
API_URL = "http://183.203.208.34:7001/api/v1/npc/chat/stream/"
HEADERS = {
    "accept": "application/json",
    "Content-Type": "application/json",
    "X-API-KEY": "zhangbi123456secure"
}

# 全局状态
GLOBAL_CTX = {
    "session_id": "",  # 将在第一次交互后更新
    "team_id": "d776880f-f8c2-4969-be32-3ac517618f2c",
    "user_id": "丽丽",
    "generated_script_id": "a39d33d3-4a97-486e-a4a5-8070c305a103"
}

# 任务列表定义
TASKS = [
    {"id": "T01_PROLOGUE", "name": "T01: 序幕 (获取门票)", "sub_task": None},
    {"id": "T02_FIRST_TRIAL", "name": "T02: 朱雀考验 (暗号)", "sub_task": None},
    {"id": "T03_TUNNEL_SEEK", "name": "T03: S1 关帝庙 (拍照)", "sub_task": "T03_S1"},
    {"id": "T03_TUNNEL_SEEK", "name": "T03: S4 西方圣境殿 (拍照)", "sub_task": "T03_S4"},
    {"id": "T04_ZAB_ARCHERY", "name": "T04: 礼射试炼 (人工确认)", "sub_task": None},
    {"id": "T05_TUNNEL_CROSS", "name": "T05: 地道穿越 (GPS)", "sub_task": None},
    {"id": "T06_SHADOW_CLOCK", "name": "T06: 影子钟 (生肖/海报)", "sub_task": None},
    {"id": "T07_CHESS_ACADEMY", "name": "T07: 棋院匾额 (解谜)", "sub_task": None},
    {"id": "T08_FINALE", "name": "T08: 终幕合成", "sub_task": None},
]

def print_stream_response(response):
    """处理流式响应并返回完整内容与数据"""
    full_content = ""
    final_data = {}
    print("NPC: ", end="", flush=True)
    
    try:
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith("data: "):
                    data_str = decoded_line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data_json = json.loads(data_str)
                        # 打印增量内容
                        if "delta" in data_json:
                            content = data_json["delta"]
                            print(content, end="", flush=True)
                            full_content += content
                        # 获取最终数据用于更新 Session
                        if "code" in data_json and "data" in data_json:
                            final_data = data_json["data"]
                    except json.JSONDecodeError:
                        pass
    except Exception as e:
        print(f"\n[流处理异常]: {e}")
    
    print("\n") # 换行
    return full_content, final_data

def chat_in_task(task_conf):
    """进入某个任务的持续对话循环"""
    # 切换任务时强制重置 session_id，清除历史上下文
    GLOBAL_CTX["session_id"] = ""
    
    print(f"\n{'='*10} 进入任务: {task_conf['name']} {'='*10}")
    print("提示: 输入 'q' 返回上一级菜单; 输入 'img' 模拟发送一张通过验证的照片")
    
    while True:
        try:
            user_input = input(f"[{task_conf['id']}] 我: ").strip()
        except EOFError:
            break

        if not user_input:
            continue
            
        if user_input.lower() == 'q':
            break

        # 构造请求
        payload = {
            "team_id": GLOBAL_CTX["team_id"],
            "user_id": GLOBAL_CTX["user_id"],
            "session_id": GLOBAL_CTX["session_id"], # 使用全局Session
            "generated_script_id": GLOBAL_CTX["generated_script_id"],
            "task_status": "in_progress",
            "task_id": task_conf["id"],
            "sub_task_id": task_conf["sub_task"],
            "history": [] # 历史由后端Redis维护
        }

        # 模拟图片指令
        if user_input.lower() == 'img':
            payload["message"] = f"我拍摄了{task_conf['name']}的照片"
            payload["image_result"] = {"success": True, "message": f"识别通过: {task_conf['name']}"}
            print(f"[系统]: 已模拟发送{task_conf['name']}的认证图片...")
        else:
            payload["message"] = user_input

        # 发起请求
        try:
            response = requests.post(API_URL, headers=HEADERS, json=payload, stream=True)
            if response.status_code != 200:
                print(f"[API Error]: Status {response.status_code} - {response.text}")
                continue
            
            _, data = print_stream_response(response)
            
            # 更新 Session ID
            if data and data.get("session_id"):
                if GLOBAL_CTX["session_id"] != data["session_id"]:
                    GLOBAL_CTX["session_id"] = data["session_id"]
                    # print(f"[Debug] Session ID updated: {GLOBAL_CTX['session_id']}")
                    
            if data and data.get("task_completed"):
                print("✨✨✨ [系统提示]: 任务判定已完成！(您可以继续对话或按q切换下一关) ✨✨✨\n")

        except Exception as e:
            print(f"[Error]: {e}")

def main():
    print("\n⚔️  张壁古堡 NPC 交互测试终端 ⚔️")
    print(f"当前剧本ID: {GLOBAL_CTX['generated_script_id']}")
    print(f"当前TeamID: {GLOBAL_CTX['team_id']}")
    print("-" * 50)
    print("背景故事: 天下动荡，义军“破晓”欲攻破“孤城”张壁。你作为内应潜入城中，")
    print("需在不同地点与接头人（NPC）联络，收集布防图残片，最终点燃烽火。")
    print("-" * 50)

    while True:
        print("\n请选择当前测试的任务节点:")
        for idx, t in enumerate(TASKS):
            status = " "
            print(f"{idx+1}. {t['name']}")
        print("0. 退出程序")

        choice = input("\n请输入编号: ").strip()
        
        if choice == '0':
            print("再见！")
            sys.exit(0)
            
        try:
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(TASKS):
                chat_in_task(TASKS[choice_idx])
            else:
                print("无效的编号，请重新输入。")
        except ValueError:
            print("请输入有效的数字编号。")

if __name__ == "__main__":
    main()
