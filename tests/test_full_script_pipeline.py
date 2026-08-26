import requests
import json
import sys

def test_npc_chat_stream():
    url = "http://183.203.208.34:7001/api/v1/npc/chat/stream/"
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
        "X-API-KEY": "zhangbi123456secure"
    }

    common_params = {
        "team_id": "d776880f-f8c2-4969-be32-3ac517618f2c",
        "user_id": "1",
        "session_id": "",
        "generated_script_id": "e259f992-d91f-43d2-8316-e34339deadf3",
        "task_status": "in_progress",
        "history": []
    }

    test_steps = [
        {
            "name": "T01_PROLOGUE - 序幕",
            "payload": {
                **common_params,
                "task_id": "T01_PROLOGUE",
                "message": "拿到了门票与任务指引，准备开始任务。",
            }
        },
        {
            "name": "T02_FIRST_TRIAL - 第一幕",
            "payload": {
                **common_params,
                "task_id": "T02_FIRST_TRIAL",
                "message": "周先生，我已经完成了九连环挑战，请您查验。",
            }
        },
        {
            "name": "T03_TUNNEL_SEEK - T03_S1 (拍照)",
            "payload": {
                **common_params,
                "task_id": "T03_TUNNEL_SEEK",
                "sub_task_id": "T03_S1",
                "message": "我已经拍好南堡门的照片了。",
                "image_result": {"success": True, "message": "识别成功"}
            }
        },
        {
            "name": "T03_TUNNEL_SEEK - T03_S2 (拍照)",
            "payload": {
                **common_params,
                "task_id": "T03_TUNNEL_SEEK",
                "sub_task_id": "T03_S2",
                "message": "我已经拍好魁星楼的照片了。",
                "image_result": {"success": True, "message": "识别成功"}
            }
        },
        {
            "name": "T03_TUNNEL_SEEK - T03_S3 (拍照)",
            "payload": {
                **common_params,
                "task_id": "T03_TUNNEL_SEEK",
                "sub_task_id": "T03_S3",
                "message": "我已经拍好北堡门的照片了。",
                "image_result": {"success": True, "message": "识别成功"}
            }
        },
        {
            "name": "T03_TUNNEL_SEEK - T03_S4 (拍照)",
            "payload": {
                **common_params,
                "task_id": "T03_TUNNEL_SEEK",
                "sub_task_id": "T03_S4",
                "message": "我已经拍好星象文化展厅的照片了。",
                "image_result": {"success": True, "message": "识别成功"}
            }
        },
        {
            "name": "T03_TUNNEL_SEEK - T03_S5_FINAL (答题)",
            "payload": {
                **common_params,
                "task_id": "T03_TUNNEL_SEEK",
                "sub_task_id": "T03_S5_FINAL",
                "message": "28",
            }
        },
        {
            "name": "T04_TUNNEL_CROSS - 地道穿越",
            "payload": {
                **common_params,
                "task_id": "T04_TUNNEL_CROSS",
                "message": "我已经穿过地道，到达出口了。",
            }
        },
        {
            "name": "T05_DYNAMIC_FOOD - 醋坊挑战",
            "payload": {
                **common_params,
                "task_id": "T05_DYNAMIC_FOOD",
                "message": "掌柜的，我已经找对醋了，并买了沙棘醋饮作为掩护。",
            }
        },
        {
            "name": "T06_DYNAMIC_TEA - 茶舍挑战",
            "payload": {
                **common_params,
                "task_id": "T06_DYNAMIC_TEA",
                "message": "主人，我已经配对完所有的茶，并下单了桑叶茶礼盒。",
            }
        },
        {
            "name": "T07_FINAL_CHESS - 子任务 S1",
            "payload": {
                **common_params,
                "task_id": "T07_FINAL_CHESS",
                "sub_task_id": "T07_S1",
                "message": "极速传输线挑战完成。",
            }
        },
        {
            "name": "T07_FINAL_CHESS - 子任务 S2",
            "payload": {
                **common_params,
                "task_id": "T07_FINAL_CHESS",
                "sub_task_id": "T07_S2",
                "message": "团体巨画创作猜测完成。",
            }
        },
        {
            "name": "T07_FINAL_CHESS - 子任务 S3",
            "payload": {
                **common_params,
                "task_id": "T07_FINAL_CHESS",
                "sub_task_id": "T07_S3",
                "message": "一圈到底挑战完成。",
            }
        },
        {
            "name": "T07_FINAL_CHESS - 子任务 S4",
            "payload": {
                **common_params,
                "task_id": "T07_FINAL_CHESS",
                "sub_task_id": "T07_S4",
                "message": "默契考验完成。",
            }
        },
        {
            "name": "T07_FINAL_CHESS - 终局棋局",
            "payload": {
                **common_params,
                "task_id": "T07_FINAL_CHESS",
                "message": "老先生，我已集齐坐标并破了这死局。",
            }
        },
        {
            "name": "T08_FINALE - 终幕合成",
            "payload": {
                **common_params,
                "task_id": "T08_FINALE",
                "message": "主持人，我已经合成五枚残片，将完整布防图送出城外了！",
            }
        }
    ]

    for step in test_steps:
        print(f"\n{'='*20} 正在测试: {step['name']} {'='*20}")
        try:
            response = requests.post(url, headers=headers, json=step['payload'], stream=True)
            if response.status_code != 200:
                print(f"❌ 请求失败，状态码: {response.status_code}")
                print(response.text)
                continue
            
            print("--- 收到流式响应 ---")
            full_content = ""
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith("data: "):
                        data_str = decoded_line[6:]
                        if data_str == "[DONE]":
                            print("\n[流结束]")
                            break
                        try:
                            data_json = json.loads(data_str)
                            if "delta" in data_json:
                                content = data_json["delta"]
                                print(content, end="", flush=True)
                                full_content += content
                            
                            # Check for new standard response format
                            if "code" in data_json and "data" in data_json:
                                print(f"\n\n[最终结果]: {json.dumps(data_json, ensure_ascii=False)}")
                            # Backward compatibility or alternative format
                            elif "final" in data_json:
                                print(f"\n\n[最终结果]: {json.dumps(data_json['final'], ensure_ascii=False)}")
                        except json.JSONDecodeError:
                            # 可能是简单的字符块
                            print(data_str, end="", flush=True)
                            full_content += data_str
            print("\n")
        except Exception as e:
            print(f"❌ 测试过程中出现异常: {e}")

if __name__ == "__main__":
    test_npc_chat_stream()
