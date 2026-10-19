"""JEV 模型在线服务（FastAPI）。

用法：
  # 默认参数启动
  .venv/bin/python infer/jev_server.py

  # 指定 checkpoint / 端口
  .venv/bin/python infer/jev_server.py --checkpoint outputs/jev_4gpu/best.pt --port 8008

  # 指定 GPU
  CUDA_VISIBLE_DEVICES=2 .venv/bin/python infer/jev_server.py

接口：
  POST /predict      表单：image=<图片路径>         返回 trend/recover 分类
  POST /predict      JSON：{"image": "...", "image_bytes": "<base64>"}
  GET  /health      服务健康检查

启动后浏览器访问 http://<host>:<port>/docs 可看交互式文档。
"""
from __future__ import annotations

import argparse
import base64
import io
import os
import sys
import tempfile

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from infer.jev_infer import JevInference

app = FastAPI(title="JEV Curve Classification Service", version="1.0")

# 全局推理器（lifespan 里初始化）
_INFER: JevInference | None = None


class PredictRequest(BaseModel):
    image: str | None = Field(None, description="图片路径（绝对路径或相对 files_root）")
    image_bytes: str | None = Field(None, description="base64 编码的图片（与 image 二选一）")


class PredictResponse(BaseModel):
    trend: dict
    recover: dict


@app.on_event("startup")
def _startup():
    global _INFER
    _INFER = app.state.infer


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _INFER is not None,
            "step": _INFER.step if _INFER else None}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if _INFER is None:
        raise HTTPException(503, "model not loaded")

    # 1) base64 图片优先
    if req.image_bytes:
        try:
            raw = base64.b64decode(req.image_bytes)
        except Exception as e:
            raise HTTPException(400, f"invalid base64: {e}")
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(raw)
            tmp_path = f.name
        try:
            return _INFER.predict_image(tmp_path)
        finally:
            os.unlink(tmp_path)

    # 2) 图片路径
    if req.image:
        try:
            return _INFER.predict_image(req.image)
        except FileNotFoundError as e:
            raise HTTPException(404, str(e))
        except Exception as e:
            raise HTTPException(500, str(e))

    raise HTTPException(400, "provide either 'image' or 'image_bytes'")


@app.post("/predict/file", response_model=PredictResponse)
def predict_file(file: UploadFile = File(...)):
    """直接上传图片文件（multipart/form-data）。"""
    if _INFER is None:
        raise HTTPException(503, "model not loaded")
    data = file.file.read()
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(data)
        tmp_path = f.name
    try:
        return _INFER.predict_image(tmp_path)
    finally:
        os.unlink(tmp_path)


def main():
    import uvicorn

    ap = argparse.ArgumentParser(description="JEV 分类服务")
    ap.add_argument("--checkpoint", default=os.path.join(REPO_ROOT, "outputs/jev_4gpu/best.pt"))
    ap.add_argument("--device", default=None, help="cuda 设备，默认自动")
    ap.add_argument("--prompt-file", default="datas/aiops/files/prompts/argos-dp-prompt.md")
    ap.add_argument("--files-root", default="datas/aiops/files")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8008)
    args = ap.parse_args()

    # 工作目录切到仓库根，保证相对路径（prompt-file/files-root/checkpoint）正确
    os.chdir(REPO_ROOT)

    print(f"[JEV server] loading checkpoint {args.checkpoint} ...", flush=True)
    infer = JevInference(args.checkpoint, device=args.device,
                         prompt_file=args.prompt_file, files_root=args.files_root)
    app.state.infer = infer
    print(f"[JEV server] model ready, listening on {args.host}:{args.port}", flush=True)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
