import io,json,sys,uuid
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'backend'))
from fastapi.testclient import TestClient
from app.main import app
from app.routes import video
from app.utils import video_utils
from app.scheduler.dispatcher import dispatcher
from app import model_registry
from app.operation import lock
client=TestClient(app)

@pytest.fixture
def uploaded():
    with (ROOT/'samples/fog.mp4').open('rb') as f:
        r=client.post('/video/upload_video',files={'file':('fog.mp4',f,'video/mp4')})
    assert r.status_code==200,r.text
    return r.json()['video_id']

def test_bad_uploads(monkeypatch):
    assert client.post('/video/upload_video',files={'file':('x.txt',b'bad','text/plain')}).status_code==415
    assert client.post('/video/upload_video',files={'file':('x.mp4',b'bad','video/mp4')}).status_code==422
    monkeypatch.setattr(video_utils,'MAX_UPLOAD_BYTES',2)
    assert client.post('/video/upload_video',files={'file':('x.mp4',b'123','video/mp4')}).status_code==413
    assert not list(video_utils.VIDEO_DIR.glob('*.upload*'))

def test_missing_model_and_auto(uploaded,monkeypatch):
    monkeypatch.delitem(dispatcher.pool._models,'aodnet')
    monkeypatch.setitem(model_registry.statuses,'aodnet',{'available':False,'reason':'test unavailable'})
    for selection in ('aodnet','auto','aodnet->prenet'):
        assert client.post(f'/video/process/{uploaded}',params={'model':selection}).status_code==503
    assert client.get(f'/video/status/{uploaded}').json()['status']=='failed'
    assert client.get(f'/video/download/{uploaded}').status_code==404
    assert client.get('/health').status_code==200
    assert client.get('/ready').status_code==503

def test_failure_removes_old_result(uploaded,monkeypatch):
    assert client.post(f'/video/process/{uploaded}',params={'model':'simple'}).status_code==200
    def fail(*a,**kw): raise RuntimeError('injected inference failure')
    monkeypatch.setattr(dispatcher,'process_frame',fail)
    r=client.post(f'/video/process/{uploaded}',params={'model':'simple'})
    assert r.status_code==500
    assert client.get(f'/video/download/{uploaded}').status_code==404
    assert not list(video_utils.PROCESSED_DIR.glob('*.writing.mp4'))
    assert not client.get(f'/video/status/{uploaded}').json()['processed']

def test_invalid_selection_and_id(uploaded):
    assert client.post(f'/video/process/{uploaded}',params={'model':'aodnet->typo'}).status_code==400
    assert client.get('/video/download/not-a-uuid').status_code==422
    assert client.post(f'/video/process/{uuid.uuid4()}').status_code==404
    assert client.post(f'/eval/evaluate/{uploaded}').status_code==409
    assert client.get(f'/video/stream/{uploaded}').status_code==409

def test_single_operation_gate(uploaded):
    with lock:
        assert client.post(f'/video/process/{uploaded}',params={'model':'simple'}).status_code==409
        assert client.post(f'/eval/evaluate/{uploaded}').status_code==409
        assert client.get('/health').status_code==200

def test_status_survives_memory_reset(uploaded):
    assert client.post(f'/video/process/{uploaded}',params={'model':'simple'}).status_code==200
    video.VIDEO_STATUS.pop(uploaded)
    assert client.get(f'/video/status/{uploaded}').json()['processed']
    assert client.get(f'/video/download/{uploaded}').status_code==200
    path=video.STATUS_DIR/f'{uploaded}.json'
    path.write_text(json.dumps({'status':'processing','processed':False}))
    assert client.get(f'/video/status/{uploaded}').json()['status']=='failed'

def test_classifier_missing_weights(tmp_path):
    from app.scheduler.weather_classifier import WeatherCNNClassifier
    with pytest.raises(FileNotFoundError): WeatherCNNClassifier(model_path=str(tmp_path/'missing.pth'))

def test_weights_never_random(tmp_path):
    from app.models.aodnet import AODNetEnhancer
    from app.models.prenet import PreNetEnhancer
    for cls in (AODNetEnhancer,PreNetEnhancer):
        with pytest.raises(FileNotFoundError): cls(weights_path=str(tmp_path/'missing.pth'))
        bad=tmp_path/'bad.pth';bad.write_bytes(b'bad')
        with pytest.raises(RuntimeError): cls(weights_path=str(bad))

def test_no_gpu_configuration(monkeypatch):
    from app import settings
    import torch
    monkeypatch.setattr(settings,'DEVICE','cuda')
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    with pytest.raises(RuntimeError): settings.device()
    monkeypatch.setattr(settings,'DEVICE','auto')
    assert settings.device()=='cpu'

def test_startup_without_assets_or_demo_directory(tmp_path):
    import os, subprocess
    environment = dict(os.environ, PYTHONPATH=str(ROOT/'backend'),
                       ROADCLEAR_MODEL_DIR=str(tmp_path/'missing-models'),
                       ROADCLEAR_DATA_DIR=str(tmp_path/'runtime'),
                       ROADCLEAR_DEMO_DIR=str(tmp_path/'missing-demo'))
    code = """from app.main import app
from fastapi.testclient import TestClient
c=TestClient(app)
assert c.get('/health').status_code==200
assert c.get('/ready').status_code==503
models=c.get('/models/status').json()['models']
assert all(not models[n]['available'] for n in ('weather','aodnet','prenet','transweather','vehicle','plate','auto'))
assert models['simple']['available']
"""
    result = subprocess.run([sys.executable,'-c',code],cwd=tmp_path,env=environment,capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stdout+result.stderr
