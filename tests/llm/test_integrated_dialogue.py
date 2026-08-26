import unittest
from unittest.mock import MagicMock, patch
import json
import sys
import os

# Ensure src is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.llm import service
from src.llm import dialogue_eval
from src.llm.dialogue_eval import PipelineOutput, RuleDecision, DialogueEvalResult, TaskState
from src.scripts.schemas import ChatRequest  # Import real schema


class TestIntegratedDialogue(unittest.TestCase):

    def setUp(self):
        # Mock DB
        self.mock_db = MagicMock()
        self.mock_script_record = MagicMock()
        
        # Sample config mimicking DB content structure for T01
        self.script_content = {
            "tasks": [
                {
                    "task_id": "T01",
                    "task_type": "NPC_INTERACTION",
                    # Add triggers for evaluation
                    "completion_criteria": {
                        "criteria": [
                            {"id": "c1", "type": "fact", "desc": "获得入城凭证", "weight": 1.0}
                        ],
                        "pass_threshold": 1.0
                    },
                    "triggers": [
                        {
                            "priority": 10,
                            "conditions": [{"type": "score", "operator": ">=", "value": 1.0}],
                            "action": {
                                "assessment": "success", 
                                "content": "好的，这是你的通关凭证，去下一个任务点吧。", 
                                "task_completed": True
                            }
                        }
                    ]
                }
            ]
        }
        self.mock_script_record.script = self.script_content
        
        # Request Object with REAL IDs and Structure
        self.chat_request = ChatRequest(
            team_id="d776880f-f8c2-4969-be32-3ac517618f2c",
            user_id="user_test_001",
            task_id="T01",
            message="我拿到了入城凭证",
            session_id="session_test_001",
            generated_script_id="5e7d7e7f-4972-44f1-95b9-1ea169243ecd",
            history=[],
            task_status="in_progress",
            sub_task_id=None,
            image_result=None
        )

    # Test 1: NPC Interaction Triggering Evaluation success
    @patch("src.llm.dialogue_eval.call_api_with_retry")
    def test_eval_triggered_success(self, mock_llm_eval):
        # Mock LLM response for signal extraction
        mock_llm_eval.return_value = json.dumps({
            "hits": {"c1": 1.0},
            "refusal": False
        })
        
        # Mock DB Query Result to return our mock record
        self.mock_db.query.return_value.filter.return_value.order_by.return_value.first.return_value = self.mock_script_record
        
        response = service.handle_dialogue_request(self.chat_request, self.mock_db)
            
        self.assertEqual(response["action"], "EVAL_RESPONSE")
        self.assertTrue(response["task_completed"])
        self.assertEqual(response["reply"], "好的，这是你的通关凭证，去下一个任务点吧。")

    # Test 2: Fallback to Legacy Process Chat (Evaluation score low or unrelated)
    @patch("src.llm.service.legacy_process_chat")
    @patch("src.llm.dialogue_eval.call_api_with_retry")
    def test_fallback_to_legacy(self, mock_llm_eval, mock_legacy_process):
        # 1. Update request message to something irrelevant to ensure "miss" is logical
        fallback_request = self.chat_request.copy()
        fallback_request.message = "今天天气不错"

        # Mock LLM response: No hits, so no rule trigger
        mock_llm_eval.return_value = json.dumps({
            "hits": {},
            "refusal": False
        })
        
        # Mock DB Query Result
        self.mock_db.query.return_value.filter.return_value.order_by.return_value.first.return_value = self.mock_script_record
        
        # Legacy returns
        mock_legacy_process.return_value = {
            "reply": "Legacy Chat Response",
            "task_completed": False,
            "action": "NONE"
        }
        
        service.handle_dialogue_request(fallback_request, self.mock_db)
        
        # Verify legacy_process_chat was called with correct arguments
        mock_legacy_process.assert_called_once()
        args, kwargs = mock_legacy_process.call_args
        
        # Verify passed arguments match request
        self.assertEqual(kwargs['message'], "今天天气不错")
        self.assertEqual(kwargs['team_id'], "d776880f-f8c2-4969-be32-3ac517618f2c")
        self.assertEqual(kwargs['generated_script_id'], "5e7d7e7f-4972-44f1-95b9-1ea169243ecd")
if __name__ == '__main__':
    unittest.main()
