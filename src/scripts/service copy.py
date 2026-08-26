from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import List, Dict, Any, TypeVar, Generic, Optional
import json
from datetime import datetime
from pathlib import Path
import os
import logging
from decimal import Decimal
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum

from pydantic import BaseModel
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy import func
from starlette.middleware.base import BaseHTTPMiddleware

from src.database.session import  get_db
from src.llm.utils import call_api_with_retry
from src.database.models import GameTeam, UserMerchant, MerchantProduct, ScriptTemplate, GeneratedScript
from src.llm.prompts.prompts import npc_dialogue_template

from fastapi import FastAPI
import time
import uuid

# Error Codes
class ErrorCode(int, Enum):
    SUCCESS = 0
    PARAM_ERROR = 40001
    BUSINESS_ERROR = 40002
    UNAUTHORIZED = 40100
    FORBIDDEN = 40300
    NOT_FOUND = 40400
    INTERNAL_ERROR = 50000

T = TypeVar("T")

class BaseResponse(BaseModel, Generic[T]):
    code: int = ErrorCode.SUCCESS.value
    message: str = "ok"
    data: Optional[T] = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Middleware
class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Process request
        try:
            response = await call_next(request)
        except Exception as e:
            logger.error(f"Middleware caught exception: {e}")
            raise e

        process_time = (time.time() - start_time) * 1000
        formatted_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Capture response body for logging
        response_body = b""
        async for chunk in response.body_iterator:
            response_body += chunk
        
        # Log
        logger.info(f"{formatted_time} [INFO]: [Response] {request.method} {request.url.path} - 响应数据: {response_body.decode('utf-8', errors='ignore')[:1000]}... - 耗时: {process_time:.2f} ms")
        
        # Reconstruct response
        return JSONResponse(
            content=json.loads(response_body),
            status_code=response.status_code,
            headers=dict(response.headers)
        )

app = FastAPI(
    title="Dynamic Script Generator",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global Exception Handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global Exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "code": ErrorCode.INTERNAL_ERROR.value,
            "message": str(exc),
            "data": None
        }
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "code": ErrorCode.BUSINESS_ERROR.value if exc.status_code == 400 else ErrorCode.INTERNAL_ERROR.value,
            "message": exc.detail,
            "data": None
        }
    )

app.add_middleware(RequestLoggingMiddleware)

router = APIRouter()

class GenerateScriptRequest(BaseModel):
    team_id: str
    style: str
    template_id: str

def get_allergens_for_team(db: Session, team_id: str) -> List[str]:
    logger.info(f"Fetching allergens for team_id: {team_id}")
    team = db.query(GameTeam).filter(GameTeam.team_id == team_id).first()
    if not team:
        logger.error(f"Team {team_id} not found in game_team")
        raise HTTPException(
            status_code=404,
            detail=f"Team {team_id} not found in game_team"
        )

    logger.info(f"Found team: {team.team_id}, aggregated_allergens: {team.aggregated_allergens}")
    return team.aggregated_allergens or []


def get_products_for_user(db: Session, team_id: str, store_name: str, item_type: str, allergens: List[str]) -> List[Dict]:
    logger.info(f"Fetching products for team_id: {team_id}, store_name: {store_name}, item_type: {item_type}, allergens: {allergens}")

    user_merchant = db.query(UserMerchant).filter(
        UserMerchant.merchant_name == store_name
    ).first()

    if not user_merchant:
        logger.error(f"Merchant not found for store {store_name}")
        raise HTTPException(
            status_code=404,
            detail=f"Merchant not found for store {store_name}"
        )

    logger.info(f"Found merchant: {user_merchant.merchant_name}, user_id: {user_merchant.user_id}")

    # Fetch all products for the store and category, sorted by stock
    all_products = db.query(MerchantProduct).filter(
        MerchantProduct.merchant_id == user_merchant.user_id,
        MerchantProduct.category == item_type
    ).order_by(MerchantProduct.stock.desc()).all()

    # Filter products by allergens in Python to ensure accuracy
    products = []
    for p in all_products:
        ingredients = p.ingredients or {}
        contains = ingredients.get("contains", [])
        may_contain = ingredients.get("may_contain", [])
        
        # Check if any allergen is present in contains or may_contain
        has_allergen = False
        for allergen in allergens:
            if allergen in contains or allergen in may_contain:
                has_allergen = True
                break
        
        if not has_allergen:
            products.append(p)

    if not products:
        logger.warning(f"No products found for store {store_name} and category {item_type} after filtering allergens {allergens}")
    else:
        logger.info(f"Filtered products: {[{'name': p.product_name, 'stock': p.stock, 'ingredients': p.ingredients} for p in products]}")

    # Select the product with the highest stock
    selected_product = products[0] if products else None

    logger.debug(f"Selected product: {selected_product}")

    return [{
        "store_name": user_merchant.merchant_name,
        "products": [
            {
                "name": selected_product.product_name,
                "type": selected_product.category,
                "stock": selected_product.stock,
                "price": selected_product.price
            }
        ] if selected_product else []
    }]

def generate_npc_dialogue(
    script_name: str,
    style: str,
    stage_name: str,
    objective_template: str,
    location: str,
    npc_role: str,
    npc_template: str,
    item_name: str,
    allergies: str
) -> str:
    system_prompt = npc_dialogue_template.format(
        script_name=script_name,
        style=style,
        stage_name=stage_name,
        objective_template=objective_template,
        location=location,
        npc_role=npc_role,
        npc_template=npc_template,
        item_name=item_name,
        allergies=allergies
    )

    messages = [
        {"role": "system", "content": system_prompt}
    ]

    return call_api_with_retry(messages) or npc_template.replace(
        "{{动态商品名称}}", item_name
    ).replace("{{用户主要过敏原}}", allergies)

def extract_script_fields(script: Dict):
    script_metadata = script["system_config"]["script_metadata"]
    mechanism = script.get("four_deities_collection", {}).get("mechanism", "")
    tasks = script["tasks"]
    
    return script_metadata, mechanism, tasks

def generate_script_content(
    team_id: str,
    style: str,
    db: Session,
    script_metadata: Dict,
    mechanism: str,
    tasks: List[Dict],
    allergy_list: List[str]
) -> Dict[str, Any]:
    final_script = {
        "script_metadata": script_metadata,
        "mechanism": mechanism,
        "total_price": 0.0,
        "tasks": []
    }

    total_price = Decimal("0.0")
    
    processed_tasks = []
    llm_futures = {}

    with ThreadPoolExecutor() as executor:
        for i, task in enumerate(tasks):
            slot = task.get("dynamic_data_slot")
            selected_item = None
            npc_dialogue = None
            stage_name = task.get("stage_name", "")
            objective_template = task.get("objective_template", "")
            location = task.get("location", "")
            npc_role = task.get("npc_role", "")
            npc_template = slot["npc_template"] if slot and "npc_template" in slot else ""
            item_name = None
            
            if slot:
                store_name = slot.get("store_name")
                item_type = slot.get("item_type")

                try:
                    #logger.info(f"Processing dynamic data slot for task_id: {task['task_id']}, store_name: {store_name}, item_type: {item_type}")
                    stores = get_products_for_user(db, team_id, store_name, item_type, allergy_list) 

                    if not stores or not stores[0]["products"]:
                        logger.warning(f"No products available for task_id: {task['task_id']}, store_name: {store_name}, item_type: {item_type}")
                    else:
                        selected_item = stores[0]["products"][0]
                        item_name = selected_item["name"]
                        if selected_item.get("price"):
                            total_price += Decimal(str(selected_item["price"]))
                        
                        #logger.info(f"Selected item for task_id: {task['task_id']}: {selected_item}")
                except Exception as e:
                    logger.error(f"Error processing dynamic data slot for task_id: {task['task_id']}: {str(e)}")
                    continue

            if slot and item_name:
                future = executor.submit(
                    generate_npc_dialogue,
                    script_name=script_metadata.get("title", "Unknown"),
                    style=style,
                    stage_name=stage_name,
                    objective_template=objective_template,
                    location=location,
                    npc_role=npc_role,
                    npc_template=npc_template,
                    item_name=item_name,
                    allergies=", ".join(allergy_list) if allergy_list else "无"
                )
                llm_futures[future] = i

            processed_tasks.append({
                "task_id": task["task_id"],
                "stage_name": stage_name,
                "location": location,
                "npc": task.get("npc"),
                "npc_role": npc_role,
                "objective_template": objective_template,
                "task_type": task.get("task_type"),
                "completion_mechanism": task.get("completion_mechanism"),
                "virtual_reward": task.get("virtual_reward"),
                "selected_item": selected_item["name"] if selected_item else None,
                "price": float(selected_item["price"]) if selected_item and selected_item.get("price") else None,
                "npc_dialogue": None,
                "sub_tasks": task.get("sub_tasks", []),
                "next_task_id": task.get("next_task_id"),
            })

        for future in as_completed(llm_futures):
            index = llm_futures[future]
            try:
                dialogue = future.result()
                processed_tasks[index]["npc_dialogue"] = dialogue
            except Exception as e:
                logger.error(f"Error generating dialogue for task index {index}: {str(e)}")

    final_script["tasks"] = processed_tasks
    final_script["total_price"] = float(total_price)

    #logger.debug(f"Final script: {final_script}")

    return final_script

def generate_full_dynamic_script(
    team_id: str,
    script_template_data: Dict,
    style: str,
    db: Session
) -> Dict[str, Any]:
    allergy_list = get_allergens_for_team(db, team_id)
    script_metadata, mechanism, tasks = extract_script_fields(script_template_data)
    
    return generate_script_content(
        team_id=team_id,
        style=style,
        db=db,
        script_metadata=script_metadata,
        mechanism=mechanism,
        tasks=tasks,
        allergy_list=allergy_list
    )


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj) 
        return super().default(obj)


@router.post("/generate-script/", response_model=BaseResponse)
async def generate_script(
    request: GenerateScriptRequest,
    db: Session = Depends(get_db)
):
    try:
        # Fetch template from DB
        template_record = db.query(ScriptTemplate).filter(
            ScriptTemplate.id == request.template_id,
            ScriptTemplate.is_active == True
        ).first()

        if not template_record:
             raise HTTPException(
                status_code=404,
                detail=f"No active script template found for id: {request.template_id}"
            )

        dynamic_script = generate_full_dynamic_script(
            team_id=request.team_id,
            script_template_data=template_record.template,
            style=request.style,
            db=db
        )

        # Save to GeneratedScript
        new_script = GeneratedScript(
            team_id=uuid.UUID(request.team_id),
            template_id=template_record.id,
            script=dynamic_script,
            status='generated'
        )
        db.add(new_script)
        db.commit()
        db.refresh(new_script)

        return BaseResponse(
            code=ErrorCode.SUCCESS.value,
            message="ok",
            data={
                "id": new_script.id
            }
        )

    except Exception as e:
        logger.exception("Script generation failed")
        raise HTTPException(
            status_code=500,
            detail=f"Script generation failed: {str(e)}"
        )


app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)