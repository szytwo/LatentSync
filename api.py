import argparse
import gc
import os
import random
from datetime import datetime
from pathlib import Path

import gradio as gr
import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import JSONResponse, PlainTextResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from omegaconf import OmegaConf
from starlette.middleware.cors import CORSMiddleware  # 引入 CORS中间件模块

from custom.AudioProcessor import AudioProcessor
from custom.TextProcessor import TextProcessor
from custom.VideoProcessor import VideoProcessor
from custom.file_utils import logging, delete_old_files_and_folders
from scripts.inference import main

result_dir = './results'
result_output_dir = f'{result_dir}/output'
CONFIG_PATH = Path("configs/unet/second_stage.yaml")
CHECKPOINT_PATH = Path("checkpoints/latentsync_unet.pt")


def process_video(
        video_path,
        audio_path,
        guidance_scale,
        inference_steps,
        seed,
):
    # Create the temp directory if it doesn't exist
    output_dir = Path(f"{result_dir}/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert paths to absolute Path objects and normalize them
    video_file_path = Path(video_path)
    video_path = video_file_path.absolute().as_posix()
    audio_path = Path(audio_path).absolute().as_posix()

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Set the output path for the processed video
    output_path = str(output_dir / f"{video_file_path.stem}_{current_time}.mp4")  # Change the filename as needed

    config = OmegaConf.load(CONFIG_PATH)

    config["run"].update(
        {
            "guidance_scale": guidance_scale,
            "inference_steps": inference_steps,
        }
    )

    # Parse the arguments
    args = create_args(video_path, audio_path, output_path, inference_steps, guidance_scale, seed)

    try:
        result = main(
            config=config,
            args=args,
        )
        clear_cuda_cache()
        print("Processing completed successfully.")
        return output_path  # Ensure the output path is returned
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        raise gr.Error(f"Error during processing: {str(e)}")


def create_args(
        video_path: str, audio_path: str, output_path: str, inference_steps: int, guidance_scale: float, seed: int
) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_ckpt_path", type=str, required=True)
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--video_out_path", type=str, required=True)
    parser.add_argument("--inference_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1247)

    return parser.parse_args(
        [
            "--inference_ckpt_path",
            CHECKPOINT_PATH.absolute().as_posix(),
            "--video_path",
            video_path,
            "--audio_path",
            audio_path,
            "--video_out_path",
            output_path,
            "--inference_steps",
            str(inference_steps),
            "--guidance_scale",
            str(guidance_scale),
            "--seed",
            str(seed),
        ]
    )


def generate_seed():
    seed = random.randint(1, 100000000)
    logging.info(f'seed: {seed}')
    return {
        "__type__": "update",
        "value": seed
    }


# 定义一个函数进行显存清理
def clear_cuda_cache():
    """
    清理PyTorch的显存和系统内存缓存。
    注意上下文，如果在异步执行，会导致清理不了
    """
    logging.info("Clearing GPU memory...")
    # 强制进行垃圾回收
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        # 重置统计信息
        torch.cuda.reset_peak_memory_stats()
        # 打印显存日志
        logging.info(f"[GPU Memory] Allocated: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")
        logging.info(f"[GPU Memory] Max Allocated: {torch.cuda.max_memory_allocated() / (1024 ** 2):.2f} MB")
        logging.info(f"[GPU Memory] Reserved: {torch.cuda.memory_reserved() / (1024 ** 2):.2f} MB")
        logging.info(f"[GPU Memory] Max Reserved: {torch.cuda.max_memory_reserved() / (1024 ** 2):.2f} MB")


# 设置允许访问的域名
origins = ["*"]  # "*"，即为所有。

app = FastAPI(docs_url=None)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # 设置允许的origins来源
    allow_credentials=True,
    allow_methods=["*"],  # 设置允许跨域的http方法，比如 get、post、put等。
    allow_headers=["*"])  # 允许跨域的headers，可以用来鉴别来源等作用。
# 挂载静态文件
app.mount("/static", StaticFiles(directory="static"), name="static")


# 使用本地的 Swagger UI 静态资源
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    logging.info("Custom Swagger UI endpoint hit")
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="Custom Swagger UI",
        swagger_js_url="/static/swagger-ui/5.9.0/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger-ui/5.9.0/swagger-ui.css",
    )


@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <!DOCTYPE html>
    <html>
        <head>
            <meta charset=utf-8>
            <title>Api information</title>
        </head>
        <body>
            <a href='./docs'>Documents of API</a>
        </body>
    </html>
    """


@app.get('/test')
async def test():
    """
    测试接口，用于验证服务是否正常运行。
    """
    return PlainTextResponse('success')


@app.post("/do")
async def do(
        video: UploadFile = File(..., description="上传的视频文件"),
        audio: UploadFile = File(..., description="上传的音频文件"),
        scale: float = Query(default=1.0, description="指导尺度（浮点数，默认值为 1.0）"),
        steps: int = Query(default=20, description="推理步数（整数，默认值为 20）"),
        fps: int = Query(default=25, description="视频帧率（整数，默认值为 25）"),
):
    """
    处理视频和音频，生成带有字幕的视频。
    返回：
        JSONResponse: 包含处理结果的 JSON 响应。
    """

    try:
        # 初始化处理器
        video_processor = VideoProcessor()
        audio_processor = AudioProcessor()

        video_upload = await video_processor.save_upload_to_video(
            upload_file=video
        )

        audio_upload = await audio_processor.save_upload_to_wav(
            upload_file=audio,
            prefix="",
            volume_multiplier=1.0,
            nonsilent=False,
            reduce_noise_enabled=False
        )
        
        video_upload = video_processor.convert_video_fps(video_upload, fps)
        video_upload = video_processor.process_video_with_audio(video_upload, audio_upload)

        seed_data = generate_seed()
        seed = seed_data["value"]

        output_path = process_video(
            video_path=video_upload,
            audio_path=audio_upload,
            guidance_scale=scale,
            inference_steps=steps,
            seed=seed
        )
        bbox_range = ''
        # 返回视频响应
        return JSONResponse({
            "errcode": 0,
            "errmsg": "ok",
            "video_path": output_path,
            "name": os.path.basename(output_path),
            "range": bbox_range
        })
    except Exception as e:
        TextProcessor.log_error(e)
        return JSONResponse({"errcode": -1, "errmsg": str(e)})
    finally:
        # 删除过期文件
        delete_old_files_and_folders(result_dir, 1)


@app.get('/download')
async def download(name: str):
    return FileResponse(path=os.path.join(result_output_dir, name), filename=name,
                        media_type='application/octet-stream')


if __name__ == "__main__":
    parserMain = argparse.ArgumentParser()
    parserMain.add_argument('--port',
                            type=int,
                            default=7810)
    argsMain = parserMain.parse_args()
    try:
        uvicorn.run(app="api:app", host="0.0.0.0", port=argsMain.port, workers=1, reload=False, log_level="info")
    except Exception as ex:
        TextProcessor.log_error(ex)
        print(ex)
        exit(0)
