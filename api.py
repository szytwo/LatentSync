import argparse
import gc
import random
import traceback
from datetime import datetime
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import JSONResponse, PlainTextResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from func_timeout import func_timeout, FunctionTimedOut
from omegaconf import OmegaConf
from starlette.middleware.cors import CORSMiddleware  # 引入 CORS中间件模块

from custom.AudioProcessor import AudioProcessor
from custom.TextProcessor import TextProcessor
from custom.VideoProcessor import VideoProcessor
from custom.file_utils import logging, delete_old_files_and_folders
from scripts.inference import main

result_dir = './results'
result_temp_dir = f'{result_dir}/temp'
result_input_dir = f'{result_dir}/input'
result_output_dir = f'{result_dir}/output'
CONFIG_PATH = Path("configs/unet/second_stage.yaml")
CHECKPOINT_PATH = Path("checkpoints/latentsync_unet.pt")


def process_video(
        video_path,
        audio_path,
        guidance_scale,
        inference_steps,
        seed,
        fps,
        max_duration
):
    args = get_main_args()

    # Create the temp directory if it doesn't exist
    output_dir = Path(result_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert paths to absolute Path objects and normalize them
    video_file_path = Path(video_path)
    video_path = video_file_path.absolute().as_posix()
    audio_path = Path(audio_path).absolute().as_posix()

    current_time = datetime.now().strftime("%H%M%S")
    # Set the output path for the processed video
    output_path = str(output_dir / f"{video_file_path.stem}_{current_time}.mp4")  # Change the filename as needed

    config = OmegaConf.load(CONFIG_PATH)

    config["run"].update(
        {
            "guidance_scale": guidance_scale,
            "inference_steps": inference_steps,
        }
    )
    config["data"].update(
        {
            "num_frames": args.num_frames,  # 输入样本连续视频帧数量，可能提升效果
            "video_fps": fps,
        }
    )
    # Parse the arguments
    args = create_args(video_path, audio_path, output_path, inference_steps, guidance_scale, seed, max_duration)

    try:
        result = main(
            config=config,
            args=args,
        )
        logging.info("Processing completed successfully.")
    except Exception as e:
        errmsg = f"Error during processing: {str(e)}"
        logging.error(errmsg)
        # 获取完整堆栈信息
        full_trace = traceback.format_exc()
        errmsg = f"{errmsg}\nFull traceback:\n{full_trace}"
        raise RuntimeError(errmsg) from e
    finally:
        clear_cuda_cache()

    return output_path  # Ensure the output path is returned


def process_video_timeout(
        video_path,
        audio_path,
        guidance_scale,
        inference_steps,
        seed,
        fps,
        max_duration
):
    """
    执行process_video，带超时，防止卡死
    """
    timeout_sec = 3600  # 超时时间

    try:
        output_path = func_timeout(
            timeout_sec,
            process_video,
            kwargs={
                "video_path": video_path,
                "audio_path": audio_path,
                "guidance_scale": guidance_scale,
                "inference_steps": inference_steps,
                "seed": seed,
                "fps": fps,
                "max_duration": max_duration
            },
        )
        return output_path
    except FunctionTimedOut:
        raise Exception(f"process_video 执行超时 {timeout_sec}s")


def create_args(
        video_path: str,
        audio_path: str,
        output_path: str,
        inference_steps: int,
        guidance_scale: float,
        seed: int,
        max_duration: int
) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_ckpt_path", type=str, required=True)
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--video_out_path", type=str, required=True)
    parser.add_argument("--inference_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1247)
    parser.add_argument("--max_duration", type=int, default=20)

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
            "--max_duration",
            str(max_duration),
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
# noinspection PyTypeChecker
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
        max_duration: int = Query(default=20, description="使用视频时长，默认值为 20秒"),
):
    """
    处理视频和音频，生成带有字幕的视频。
    返回：
        JSONResponse: 包含处理结果的 JSON 响应。
    """

    args = get_main_args()
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
            reduce_noise_enabled=False,
            delay=int(args.num_frames / fps * 1000) + 200
        )

        video_upload, fps = video_processor.convert_video_fps(video_upload, fps)
        # video_upload = video_processor.process_video_with_audio(video_upload, audio_upload)

        seed_data = generate_seed()
        seed = seed_data["value"]

        output_path = process_video_timeout(
            video_path=video_upload,
            audio_path=audio_upload,
            guidance_scale=scale,
            inference_steps=steps,
            seed=seed,
            fps=fps,
            max_duration=max_duration
        )
        bbox_range = ''
        # 返回视频响应
        return JSONResponse({
            "errcode": 0,
            "errmsg": "ok",
            "name": output_path,
            "range": bbox_range
        })
    except Exception as e:
        TextProcessor.log_error(e)
        return JSONResponse({"errcode": -1, "errmsg": str(e)})
    finally:
        # 删除过期文件
        delete_old_files_and_folders(result_dir, 1)


@app.get('/download')
async def download(
        name: str = Query(..., description="输入文件路径"),
):
    file_name = Path(name).name
    return FileResponse(path=name, filename=file_name, media_type='application/octet-stream')


def get_main_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=7810)
    # 设置显存比例限制（浮点类型，默认值为 0）
    parser.add_argument("--cuda_memory", type=float, default=0)
    parser.add_argument("--num_frames", type=int, default=16)

    return parser.parse_args()  # ✅ 每次调用都解析参数


if __name__ == "__main__":
    argsMain = get_main_args()
    # 设置显存比例限制
    if argsMain.cuda_memory > 0:
        logging.info(f"cuda_memory: {argsMain.cuda_memory}")
        torch.cuda.set_per_process_memory_fraction(argsMain.cuda_memory)

    try:
        # 删除临时文件
        delete_old_files_and_folders(result_dir, 0)

        uvicorn.run(app="api:app", host="0.0.0.0", port=argsMain.port, workers=1, reload=False, log_level="info")
    except Exception as ex:
        TextProcessor.log_error(ex)
        print(ex)
        exit(0)
