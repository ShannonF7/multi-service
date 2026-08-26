import requests
import json
import uuid

# Configuration
BASE_URL = "http://localhost:8000/api/v1"
TEAM_ID = "d776880f-f8c2-4969-be32-3ac517618f2c"
USER_ID = "player_123"
TASK_ID = "T03_TUNNEL_SEEK" # Change this to test different tasks
GENERATED_SCRIPT_ID = "440613a7-0acf-461e-bfb3-4d870393ed0c"
API_KEY = "zhangbi123456secure"

HEADERS = {
    "X-API-KEY": API_KEY,
    "Content-Type": "application/json"
}


def chat_loop():
    print(f"Starting chat session for Task: {TASK_ID}")
    print("Type 'quit' to exit, 'clear' to reset session.")
    
    # Client-side session generation
    session_id = str(uuid.uuid4())
    print(f"[System]: Initialized Session ID: {session_id}")
    
    sub_task_id = None # Start with no sub-task
    task_status = "in_progress"

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['quit', 'exit']:
            break
        if user_input.lower() == 'clear':
            session_id = str(uuid.uuid4())
            print(f"Session cleared. New Session ID: {session_id}")
            continue

        # Construct payload
        payload = {
            "team_id": TEAM_ID,
            "user_id": USER_ID,
            "task_id": TASK_ID,
            "message": user_input,
            "session_id": session_id,
            "generated_script_id": GENERATED_SCRIPT_ID,
            "task_status": task_status,
            "sub_task_id": sub_task_id
        }

        try:
            response = requests.post(f"{BASE_URL}/npc/chat/", json=payload, headers=HEADERS)
            if response.status_code != 200:
                print(f"Request failed: {response.status_code} - {response.text}")
                continue
                
            data = response.json()
            
            if data['code'] != 0:
                print(f"Error: {data['message']}")
                continue

            result = data['data']
            reply = result['reply']
            action = result['action']
            
            print(f"NPC: {reply}")
            if action != "NONE":
                print(f"[System Action]: {action} Data: {result.get('action_data')}")

            # Update State (Simulation)
            if result.get('task_completed'):
                print(">>> Task Completed! <<<")
                task_status = "completed"
            
        except Exception as e:
            print(f"Request failed: {e}")

def assistant_loop():
    print(f"Starting Assistant Chat for Task: {TASK_ID}")
    print("Type 'quit' to exit, 'intro' to test intro, 'outro' to test outro.")
    
    session_id = str(uuid.uuid4())
    print(f"[System]: Initialized Session ID: {session_id}")
    
    task_status = "in_progress"

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['quit', 'exit']:
            break
        
        if user_input.lower() == 'intro':
            task_status = "starting"
            user_input = "开始任务" # Dummy message
        elif user_input.lower() == 'outro':
            task_status = "completed"
            user_input = "任务完成"
        else:
            task_status = "in_progress"

        # Construct payload
        payload = {
            "team_id": TEAM_ID,
            "user_id": USER_ID,
            "task_id": TASK_ID,
            "message": user_input,
            "session_id": session_id,
            "generated_script_id": GENERATED_SCRIPT_ID,
            "task_status": task_status
        }

        try:
            response = requests.post(f"{BASE_URL}/assistant/chat/", json=payload, headers=HEADERS)
            if response.status_code != 200:
                print(f"Request failed: {response.status_code} - {response.text}")
                continue
                
            data = response.json()
            
            if data['code'] != 0:
                print(f"Error: {data['message']}")
                continue

            result = data['data']
            reply = result['reply']
            
            print(f"Assistant: {reply}")
            
        except Exception as e:
            print(f"Request failed: {e}")

if __name__ == "__main__":
    mode = input("Select mode (1: NPC Chat, 2: Assistant Chat): ")
    if mode == "2":
        assistant_loop()
    else:
        chat_loop()
