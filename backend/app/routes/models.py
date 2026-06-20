from typing import Dict

from fastapi import APIRouter, HTTPException

from ..models.aodnet import AODNetEnhancer
from ..models.prenet import PreNetEnhancer
from ..scheduler.dispatcher import dispatcher

router = APIRouter()


@router.get("/pool")
async def list_model_pool() -> Dict[str, Dict[str, str]]:
    """
    列出当前模型池中的所有模型及其类型。
    """
    return {"models": dispatcher.pool.list_models()}


@router.post("/add/{name}")
async def add_model(name: str, model_type: str) -> Dict[str, str]:
    """
    向模型池动态添加模型。

    :param name: 模型在池中的标识名，例如 "aodnet2"
    :param model_type: 模型类型，当前支持 "aodnet" / "prenet" / "simple"
    """
    model_type = model_type.lower()
    if model_type == "aodnet":
        model = AODNetEnhancer()
    elif model_type == "prenet":
        model = PreNetEnhancer()
    elif model_type == "simple":
        # simple 为轻量恒等/轻量增强，由调度器内部实现，这里无需真正模型实例
        model = object()
    else:
        raise HTTPException(status_code=400, detail=f"不支持的模型类型: {model_type}")

    dispatcher.pool.add_model(name, model)
    return {"status": "ok", "name": name, "type": model_type}


@router.delete("/remove/{name}")
async def remove_model(name: str) -> Dict[str, str]:
    """
    从模型池中移除指定模型。
    """
    dispatcher.pool.remove_model(name)
    return {"status": "ok", "removed": name}

