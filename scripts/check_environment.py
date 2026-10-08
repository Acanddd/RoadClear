import hashlib,json,platform,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backend'))
from app import settings
import torch,onnxruntime,imageio_ffmpeg
manifest=json.loads((ROOT/'docs/model-manifest.json').read_text())
assets=[]
for record in manifest:
    p=settings.model_path(record['id'])
    digest=hashlib.file_digest(p.open('rb'),'sha256').hexdigest() if p.is_file() else None
    assets.append({'id':record['id'],'exists':p.is_file(),'sha256_matches':digest==record['sha256']})
result={'python':sys.version,'platform':platform.platform(),'device':settings.DEVICE,'torch':torch.__version__,
        'cuda_available':torch.cuda.is_available(),'onnx_providers':onnxruntime.get_available_providers(),
        'ffmpeg':imageio_ffmpeg.get_ffmpeg_exe(),'ffmpeg_version':imageio_ffmpeg.get_ffmpeg_version(),
        'node':shutil.which('node'),'docker':shutil.which('docker'),'models':assets}
print(json.dumps(result,indent=2))
if not all(a['sha256_matches'] for a in assets):
    raise SystemExit(1)
