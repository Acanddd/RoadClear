"""Run against TestClient by default, or --base-url for native/container validation."""
import argparse, hashlib, json, os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backend'))
import cv2
import numpy as np
import httpx
import onnxruntime as ort
from app.utils.video_utils import write_video, video_info

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-url')
    parser.add_argument('--report', default='runtime/validation.json')
    args = parser.parse_args()
    if args.base_url:
        client = httpx.Client(base_url=args.base_url, timeout=300)
    else:
        from fastapi.testclient import TestClient
        from app.main import app
        client = TestClient(app)
    report = {'mode': args.base_url or 'TestClient', 'device': 'cpu', 'cases': []}
    assert client.get('/health').status_code == 200
    report['models'] = client.get('/models/status').json()['models']
    assert client.get('/ready').status_code == (200 if all(report['models'][n]['available'] for n in ('aodnet','prenet','transweather','weather','vehicle','plate')) else 503)
    for weather, model in [('fog','aodnet'),('rain','prenet'),('snow','transweather')]:
        sample = ROOT / 'samples' / f'{weather}.mp4'
        if not sample.exists():
            image = cv2.imread(str(ROOT / 'frontend/image_display' / f'{weather}1.jpg'))
            if image is None:
                raise RuntimeError('Missing fixed sample image')
            frame = cv2.resize(image, (128,96))
            write_video(sample, (frame.copy() for _ in range(3)), fps=6)
        if not report['models'][model]['available']:
            report['cases'].append({'model':model, 'status':'disabled', 'reason':report['models'][model]['reason']})
            continue
        start=time.perf_counter()
        with sample.open('rb') as f:
            response=client.post('/video/upload_video',files={'file':(sample.name,f,'video/mp4')})
        assert response.status_code==200,response.text
        vid=response.json()['video_id']
        response=client.post(f'/video/process/{vid}',params={'model':model})
        assert response.status_code==200,response.text
        result=response.json()
        assert result['processed'] and result['frames_processed']==3 and model in result['model_chain']
        data=client.get(result['download_url'])
        assert data.status_code==200 and len(data.content)>100
        range_result=client.get(f'/video/download/{vid}',headers={'Range':'bytes=0-99'})
        assert range_result.status_code==206 and len(range_result.content)==100,range_result.status_code
        original=client.get(f'/video/original/{vid}',headers={'Range':'bytes=0-99'})
        assert original.status_code==206
        local=ROOT/'runtime'/f'validated-{weather}.mp4';local.write_bytes(data.content)
        info=video_info(local)
        assert info['frames']==3 and abs(info['fps']-6)<.1
        cap=cv2.VideoCapture(str(local));ok,frame=cap.read();cap.release()
        assert ok and frame.shape==(96,128,3) and np.isfinite(frame).all()
        cv2.imwrite(str(ROOT/'runtime'/f'validated-{weather}.png'),frame)
        case={'model':model,'video_id':vid,'status':'passed','seconds':round(time.perf_counter()-start,3),'sample_sha256':hashlib.sha256(sample.read_bytes()).hexdigest(),'output_sha256':hashlib.sha256(data.content).hexdigest(),'video_info':info}
        if report['models']['vehicle']['available'] and report['models']['plate']['available']:
            evaluation=client.post(f'/eval/evaluate/{vid}')
            assert evaluation.status_code==200,evaluation.text
            case['evaluation']=evaluation.json()['metrics']
            assert 'html_report' in evaluation.json()
        report['cases'].append(case)
        # Repeat with the legacy snow alias and post-processing, avoiding stale output.
        again=client.post(f'/video/process/{vid}',params={'model':'hdcwnet' if model=='transweather' else model,'post_process':True})
        assert again.status_code==200,again.text
        assert client.get(result['download_url']).status_code==404
        assert client.get(again.json()['download_url']).status_code==200
        assert client.get(f'/video/status/{vid}').json()['post_processed']
    path=Path(args.report);path=path if path.is_absolute() else ROOT/path
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Validation report:',path)
    client.close()

if __name__=='__main__':
    main()
