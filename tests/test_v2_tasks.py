import requests
import json
import sys
import os

# 配置信息（按需修改）
API_URL = "http://183.203.208.34:7001/api/v1/npc/chat/stream/"
HEADERS = {
    "accept": "application/json",
    "Content-Type": "application/json",
    "X-API-KEY": "zhangbi123456secure"
}

# 全局状态
GLOBAL_CTX = {
    "session_id": "",
    "team_id": "d776880f-f8c2-4969-be32-3ac517618f2c",
    "user_id": "测试玩家",
    "generated_script_id": "feb554d2-d9d3-4f25-8dbc-b0447b5c4d5b"
}

# 从 v2.json 自动加载任务列表
V2_JSON_PATH = os.path.join("src", "llm", "prompts", "v2.json")

def load_tasks_from_v2(path=V2_JSON_PATH):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        tasks = data.get('tasks', [])
        task_list = []
        for t in tasks:
            # 取第一个子任务 id（如果存在）作为示例
            sub = None
            subs = t.get('sub_tasks') or []
            if subs and len(subs) > 0:
                sub = subs[0].get('sub_task_id')
            task_list.append({
                'id': t.get('task_id'),
                'name': f"{t.get('task_id')} - {t.get('stage_name')}",
                'sub_task': sub
            })
        return task_list
    except Exception as e:
        print(f"加载 v2.json 失败: {e}")
        return []

TASKS = load_tasks_from_v2()


def print_stream_response(response):
    """处理流式响应并返回完整内容与最终 data （如果有）"""
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
                        # 捕获最终返回的 code/data
                        if isinstance(data_json, dict) and data_json.get("code") is not None and data_json.get("data") is not None:
                            final_data = data_json["data"]
                    except json.JSONDecodeError:
                        pass
    except Exception as e:
        print(f"\n[流处理异常]: {e}")
    print("\n")
    return full_content, final_data


def chat_in_task(task_conf):
    """进入某个任务的交互循环。
    支持命令：
      - q: 退出回菜单
      - img: 模拟上传通过验证的图片（将 image_result.success=True）
      - complete: 模拟系统将 task_status 置为 completed（用于 STAFF_CONFIRM/GPS_CHECK/COMBINE_SUCCESS 测试）
    """
    GLOBAL_CTX['session_id'] = ""  # 切换任务时重置 session

    print(f"\n{'='*10}  进入任务: {task_conf['name']} {'='*10}")
    print("命令提示: 输入 'q' 返回上一级; 输入 'img' 模拟通过图片验证; 输入 'complete' 模拟系统置为 completed")

    while True:
        try:
            user_input = input(f"[{task_conf['id']}] 我: ").strip()
        except EOFError:
            break

        if not user_input:
            continue
        if user_input.lower() == 'q':
            break

        # 默认 task_status 为 in_progress，除非用户用 complete 命令
        simulated_task_status = 'in_progress'
        payload = {
            "team_id": GLOBAL_CTX['team_id'],
            "user_id": GLOBAL_CTX['user_id'],
            "session_id": GLOBAL_CTX['session_id'],
            "generated_script_id": GLOBAL_CTX['generated_script_id'],
            "task_status": simulated_task_status,
            "task_id": task_conf['id'],
            "sub_task_id": task_conf['sub_task'],
            "history": []
        }

        if user_input.lower() == 'img':
            payload['message'] = f"我拍摄了{task_conf['name']}的照片"
            payload['image_result'] = {"success": True, "message": f"识别通过: {task_conf['name']}"}
            print(f"[系统]: 已模拟发送{task_conf['name']}的认证图片...")
        elif user_input.lower() == 'complete':
            # 模拟系统确认完成（用于 STAFF_CONFIRM / GPS_CHECK / COMBINE_SUCCESS 等）
            payload['message'] = f"系统标记为已完成（模拟）"
            payload['task_status'] = 'completed'
            payload['image_result'] = payload.get('image_result', {})
            print("[系统]: 已模拟将 task_status 置为 completed（仅用于测试）")
        else:
            payload['message'] = user_input

        try:
            response = requests.post(API_URL, headers=HEADERS, json=payload, stream=True, timeout=30)
            if response.status_code != 200:
                print(f"[API Error]: Status {response.status_code} - {response.text}")
                continue

            _, data = print_stream_response(response)

            # 更新 Session ID
            if data and data.get('session_id'):
                if GLOBAL_CTX['session_id'] != data['session_id']:
                    GLOBAL_CTX['session_id'] = data['session_id']

            # 显示是否系统已判定完成
            if data and data.get('task_completed'):
                print("✨✨✨ [系统提示]: 任务判定已完成！(您可以继续对话或按q切换下一关) ✨✨✨\n")

        except Exception as e:
            print(f"[请求异常]: {e}")


def main():
    if not TASKS:
        print("未能从 v2.json 加载任务，检查路径 src/llm/prompts/v2.json 是否存在。")
        sys.exit(1)

    print("\n🧭 V2 剧本自动化测试终端 🧭")
    print("说明：脚本会从 src/llm/prompts/v2.json 读取任务清单并提供交互测试。")

    while True:
        print("\n请选择测试的任务节点:")
        for idx, t in enumerate(TASKS):
            print(f"{idx+1}. {t['name']} (sub: {t['sub_task']})")
        print("0. 退出程序")

        choice = input("\n请输入编号: ").strip()
        if choice == '0':
            print("退出。")
            sys.exit(0)
        try:
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(TASKS):
                chat_in_task(TASKS[choice_idx])
            else:
                print("无效编号，请重试。")
        except ValueError:
            print("请输入有效的数字编号。")

if __name__ == '__main__':
    main()
