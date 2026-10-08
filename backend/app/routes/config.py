"""
参数配置路由
提供前端参数配置和管理接口
"""
from typing import Dict, Any

from fastapi import APIRouter, HTTPException

from ..config_manager import ParameterManager, ParameterConfig

from ..config_manager import get_parameter_manager

router = APIRouter()


@router.get("/config", response_model=ParameterConfig)
async def get_config() -> ParameterConfig:
    """
    获取当前参数配置
    """
    return get_parameter_manager().get_config()


@router.post("/config")
async def update_config(updates: Dict[str, Any]) -> ParameterConfig:
    """
    更新参数配置
    """
    try:
        if "models" in updates:
            current = get_parameter_manager().get_config().models.model_dump()
            proposed = dict(updates["models"])
            if "transweather" in proposed:
                proposed["hdcwnet"] = proposed.pop("transweather")
            if any(current.get(k) != v for k, v in proposed.items()):
                raise ValueError("Model runtime settings are fixed in the validated local profile; configure and restart")
        return get_parameter_manager().update_config(updates)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"参数更新失败: {str(e)}")


@router.post("/config/reset")
async def reset_config() -> ParameterConfig:
    """
    重置参数配置为默认值
    """
    return get_parameter_manager().reset_to_defaults()


@router.get("/config/schema")
async def get_config_schema() -> Dict[str, Any]:
    """
    获取参数配置的 JSON Schema，用于前端表单验证
    """
    return ParameterConfig.schema()
