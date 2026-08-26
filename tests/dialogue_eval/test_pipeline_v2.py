import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.llm.dialogue_eval import run_task_pipeline, TaskState

def test_pipeline_success():
    print("Testing T01_PROLOGUE Success Case...")
    user_text = "我拿到门票跟任务指引了。"
    
    try:
        output = run_task_pipeline(
            task_id="T01_PROLOGUE",
            user_text=user_text,
            prev_state=TaskState.IDLE
        )
        
        print(f"Input: {user_text}")
        print(f"Score: {output.eval_result.score}")
        print(f"Assessment: {output.rule.assessment}")
        print(f"Message: {output.rule.message}")
        print(f"Next State: {output.next_state}")
        
    except Exception as e:
        print(f"Error during pipeline execution: {e}")

def test_pipeline_info():
    print("Testing T01_PROLOGUE Info Case...")
    user_text = "我有门票了，但是任务指引是什么？"
    
    try:
        output = run_task_pipeline(
            task_id="T01_PROLOGUE",
            user_text=user_text,
            prev_state=TaskState.IDLE
        )
        
        print(f"Input: {user_text}")
        print(f"Score: {output.eval_result.score}")
        print(f"Assessment: {output.rule.assessment}")
        print(f"Message: {output.rule.message}")
        print(f"Next State: {output.next_state}")
        
    except Exception as e:
        print(f"Error during pipeline execution: {e}")

def test_pipeline_warning():
    print("Testing T01_PROLOGUE Warning Case...")
    user_text = "我已经拿到了门票和任务指引，你个大傻子，这都听不懂吗？"
    
    try:
        output = run_task_pipeline(
            task_id="T01_PROLOGUE",
            user_text=user_text,
            prev_state=TaskState.IDLE
        )
        
        print(f"Input: {user_text}")
        print(f"Score: {output.eval_result.score}")
        print(f"Assessment: {output.rule.assessment}")
        print(f"Message: {output.rule.message}")
        print(f"Next State: {output.next_state}")
        
    except Exception as e:
        print(f"Error during pipeline execution: {e}")

def test_pipeline_refusal():
    print("\nTesting T01_PROLOGUE Refusal Case...")
    user_text = "我不想做这个任务，太麻烦了。"
    
    try:
        output = run_task_pipeline(
            task_id="T01_PROLOGUE",
            user_text=user_text,
            prev_state=TaskState.IDLE
        )
        
        print(f"Input: {user_text}")
        print(f"Refusal: {output.eval_result.refusal}")
        print(f"Assessment: {output.rule.assessment}")
        print(f"Message: {output.rule.message}")
        print(f"Next State: {output.next_state}")
        
    except Exception as e:
        print(f"Error during pipeline execution: {e}")


def test_pipeline_pass():
    print("Testing T01_PROLOGUE Pass Case...")
    user_text = "什么门票任务指引？我根本没听说过。"
    
    try:
        output = run_task_pipeline(
            task_id="T01_PROLOGUE",
            user_text=user_text,
            prev_state=TaskState.IDLE
        )
        
        print(f"Input: {user_text}")
        print(f"Score: {output.eval_result.score}")
        print(f"Assessment: {output.rule.assessment}")
        print(f"Message: {output.rule.message}")
        print(f"Next State: {output.next_state}")
        
    except Exception as e:
        print(f"Error during pipeline execution: {e}")

def test_pipeline_pass1():
    print("Testing T01_PROLOGUE Pass1 Case...")
    user_text = "拿到了"
    
    try:
        output = run_task_pipeline(
            task_id="T01_PROLOGUE",
            user_text=user_text,
            prev_state=TaskState.IDLE
        )
        
        print(f"Input: {user_text}")
        print(f"Score: {output.eval_result.score}")
        print(f"Assessment: {output.rule.assessment}")
        print(f"Message: {output.rule.message}")
        print(f"Next State: {output.next_state}")
        
    except Exception as e:
        print(f"Error during pipeline execution: {e}")

if __name__ == "__main__":
    test_pipeline_success()
    test_pipeline_info()
    test_pipeline_warning()
    test_pipeline_refusal()
    test_pipeline_pass()
    test_pipeline_pass1()
