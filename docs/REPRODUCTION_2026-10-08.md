# Local functional recheck — 2026-10-08

This snapshot was rechecked on Windows with Python 3.12 and CPU inference before repository archival.

- Environment check passed. All six model files exist and match the model-manifest SHA-256 values.
- Existing end-to-end validation passed through FastAPI TestClient for AOD-Net, PReNet, and TransWeather: upload, enhancement, evaluation report, download, HTTP Range, output frame count/FPS, repeat processing and post-processing.
- The fixed inputs contain three frames at 128×96 and 6 FPS. They verify the functional workflow, not research accuracy or realistic throughput.
- Regression tests: 10 passed. The default old pytest temporary directory produced a Windows permission error; rerunning with a fresh project-local directory succeeded.
- Vue production build passed. The main JavaScript chunk remains approximately 1.17 MB; Vite reported its existing chunk-size warning.

Commands used:

```powershell
./scripts/check.ps1
./scripts/verify.ps1
./.venv-local/Scripts/python.exe -m pytest backend/tests/test_local_profile.py -q --basetemp=runtime/pytest-review-20261008
# From frontend/vue, using the installed Node executable:
node node_modules/vite/bin/vite.js build
```

The end-to-end portion of verify.ps1 passed; its first regression run failed on the old temporary directory. The separate regression command above passed all tests. Frontend build and model validation required execution outside the restricted tool sandbox.

Raw end-to-end results are in ignored runtime/validation.json. No browser interaction, native HTTP server startup, Docker/GPU validation, retraining, or reproduction of thesis metrics was performed in this recheck. Refer to VALIDATION_REPORT.md for earlier native HTTP/browser validation and current deployment limitations.

Large weights, virtual environments, runtime outputs and local secrets remain excluded from Git. This snapshot is archived on a separate Git branch so the original graduation-project master branch remains available.
