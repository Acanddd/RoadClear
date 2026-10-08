from fastapi import APIRouter, HTTPException
from .. import model_registry
from ..scheduler.dispatcher import dispatcher
router = APIRouter()
@router.get('/pool')
def pool():
    return {'models': dispatcher.pool.list_models(), 'status': model_registry.capabilities()}
@router.get('/status')
def status():
    return {'models': model_registry.capabilities()}
@router.post('/add/{name}')
def add(name: str, model_type: str):
    raise HTTPException(409, 'Local profile uses fixed validated models; change configuration and restart')
@router.delete('/remove/{name}')
def remove(name: str):
    raise HTTPException(409, 'Use ROADCLEAR_DISABLED_MODELS and restart')
