import json
import math
import os
import subprocess

import cv2
from fastapi import UploadFile

from custom.TextProcessor import TextProcessor
from custom.file_utils import logging, add_suffix_to_filename


class VideoProcessor:
    def __init__(self, input_dir="results/input", output_dir="results/output"):
        """
        初始化视频处理器，设置临时文件目录。
        :param input_dir: 输入文件目录
        :param output_dir: 输出文件目录
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

    async def save_upload_to_video(self, upload_file: UploadFile):
        """
        保存上传的文件到本地并返回路径。
        :param upload_file: FastAPI 的上传文件对象
        :return: 保存后的文件路径
        """
        # 构建保存路径
        upload_path = os.path.join(self.input_dir, upload_file.filename)
        # 如果同名文件已存在，先删除
        if os.path.exists(upload_path):
            os.remove(upload_path)

        logging.info(f"接收上传 {upload_file.filename} 请求 {upload_path}")

        try:
            # 异步保存上传的文件内容
            with open(upload_path, "wb") as f:
                f.write(await upload_file.read())  # 异步读取并写入文件

            return upload_path
        except Exception as e:
            raise Exception(f"{upload_file.filename} 视频文件保存失败: {str(e)}")
        finally:
            await upload_file.close()  # 显式关闭上传文件

    @staticmethod
    def convert_video_fps(video_path: str, target_fps: int = 25):
        """ 将视频转换为 指定 FPS """
        # 检查视频帧率
        original_fps = VideoProcessor.get_video_frame_rate(video_path)

        if original_fps != target_fps:
            logging.info(f"视频帧率为 {original_fps} FPS，转换为 {target_fps} FPS")
            converted_video_path = add_suffix_to_filename(video_path, f"_{target_fps}")

            # 使用 FFmpeg 转换帧率
            try:
                # NVIDIA 编码器 codec="h264_nvenc"    CPU编码 codec="libx264"
                # 创建 FFmpeg 命令来合成视频
                cmd = [
                    "ffmpeg",
                    "-i", video_path,
                    "-r", f"{target_fps}",  # 设置输出帧率
                    "-c:v", "libx264",  # 使用 libx264 编码器
                    "-crf", "18",  # 设置压缩质量
                    "-preset", "slow",  # 设置编码速度/质量平衡
                    "-c:a", "aac",  # 设置音频编码器
                    "-b:a", "192k",  # 设置音频比特率
                    "-ar", "44100",
                    "-ac", "2",
                    "-y",
                    converted_video_path
                ]
                # 执行 FFmpeg 命令
                subprocess.run(cmd, capture_output=True, text=True, check=True)

                logging.info(f"视频转换完成: {converted_video_path}")
                return converted_video_path, target_fps
            except subprocess.CalledProcessError as e:
                # 捕获任何在处理过程中发生的异常
                ex = Exception(f"Error ffmpeg: {e.stderr}")
                TextProcessor.log_error(ex)
                return None, None
        else:
            logging.info(f"视频帧率已经是 {target_fps} FPS，无需转换")
            return video_path, original_fps

    @staticmethod
    def get_media_metadata(media_path):
        """
        使用 ffprobe 提取媒体文件的元数据，并以 JSON 格式返回。
        """
        cmd = [
            "ffprobe",
            "-i", media_path,
            "-show_streams",
            "-show_format",
            "-print_format", "json",
            "-hide_banner",
            "-loglevel", "error"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        try:
            metadata = json.loads(result.stdout)
        except json.JSONDecodeError:
            metadata = {}

        return metadata

    @staticmethod
    def get_duration(media_path):
        """利用 ffprobe 获取多媒体文件时长（单位：秒）"""
        metadata = VideoProcessor.get_media_metadata(media_path)
        # 从 format 元数据中提取时长
        duration = float(metadata.get("format", {}).get("duration", "0.0"))

        return duration

    @staticmethod
    def get_video_metadata(media_path):
        """获取视频文件的元数据信息"""
        metadata = VideoProcessor.get_media_metadata(media_path)
        # 查找第一个视频流
        video_stream = next((stream for stream in metadata.get("streams", []) if stream.get("codec_type") == "video"),
                            None)
        if not video_stream:
            raise ValueError("未找到视频流")

        return video_stream

    @staticmethod
    def get_video_frame_rate(media_path):
        """获取视频文件的帧率"""
        video_metadata = VideoProcessor.get_video_metadata(media_path)
        # 获取 r_frame_rate
        r_frame_rate = video_metadata.get("r_frame_rate", "0/1")
        # 计算帧率
        num, denom = map(int, r_frame_rate.split('/'))
        frame_rate = num / denom if denom != 0 else 0

        return frame_rate

    @staticmethod
    def process_video_with_audio(video_path: str, audio_path: str):
        # 获取视频和音频时长
        video_duration = VideoProcessor.get_duration(video_path)
        audio_duration = VideoProcessor.get_duration(audio_path)

        logging.info(f"视频时长: {video_duration} 秒，音频时长: {audio_duration} 秒")

        if video_duration == audio_duration:
            return video_path

        output_path = add_suffix_to_filename(video_path, "_with")

        if video_duration > audio_duration:
            logging.info(f"视频时长大于或等于音频，直接截取视频")

            cmd_trim = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-i", audio_path,
                "-c:v", "libx264",
                "-crf", "18",  # 设置压缩质量
                "-preset", "slow",  # 设置编码速度/质量平衡
                "-c:a", "aac",
                "-b:a", "192k",  # 设置音频比特率
                "-ar", "44100",
                "-ac", "2",
                "-t", str(audio_duration),
                output_path
            ]
            subprocess.run(cmd_trim, capture_output=True, text=True, check=True)
        else:
            logging.info(f"视频时长小于音频，正倒序重复拼接......")
            # 要剪掉的时长，单位：毫秒
            offset_ms = 100
            # 转换为秒（浮点数）
            offset_seconds = offset_ms / 1000.0
            # 对于 ffmpeg -ss 参数，可以直接使用字符串形式的秒数
            offset = str(offset_seconds)  # 例如 "0.5"
            # 生成视频倒序版本（忽略音频）
            reversed_video = add_suffix_to_filename(video_path, "_reversed")
            cmd_reverse = [
                "ffmpeg", "-y",
                "-ss", offset,  # 先去掉前面的 offset_seconds
                "-i", video_path,
                "-vf", "reverse",
                "-an",  # 不处理音频
                "-crf", "18",  # 设置压缩质量
                "-preset", "slow",  # 设置编码速度/质量平衡
                reversed_video
            ]
            subprocess.run(cmd_reverse, capture_output=True, text=True, check=True)
            # 计算每个视频输入的有效时长（去掉前面的 offset）
            effective_video_duration = video_duration - offset_seconds
            # 每个正序+倒序周期的时长为两个有效视频时长之和
            cycle_duration = effective_video_duration * 2
            # 根据音频时长和每个周期的时长计算需要的循环次数
            cycles = math.ceil(audio_duration / cycle_duration)
            logging.info(f"需要循环 {cycles} 个周期")
            # 构造 concat 过滤器，只拼接视频流（注意，这里每个输入视频已经经过 -ss 剪切）
            concat_filter = "".join(
                f"[{i}:v:0]" for i in range(cycles * 2)
            ) + f"concat=n={cycles * 2}:v=1[outv]"
            # 构造交替顺序的输入列表，并在每个视频输入前加上 -ss 参数
            inputs = []
            for i in range(cycles * 2):
                if i % 2 == 0:
                    # 正序视频
                    inputs.extend(["-ss", offset, "-i", video_path])
                else:
                    # 倒序视频（已经在生成时去掉了前面部分，不需要再加 -ss）
                    inputs.extend(["-i", reversed_video])
            # 拼接命令：注意这里音频输入位于所有视频输入之后，其索引为 cycles * 2
            cmd_concat = [
                "ffmpeg", "-y",
                *inputs,
                "-i", audio_path,
                "-filter_complex", concat_filter,
                "-map", "[outv]",  # 映射拼接后的视频流
                "-map", f"{cycles * 2}:a:0",  # 映射音频文件中的音频流
                "-c:v", "libx264",
                "-crf", "18",  # 设置压缩质量
                "-preset", "slow",  # 设置编码速度/质量平衡
                "-c:a", "aac",
                "-b:a", "192k",  # 设置音频比特率
                "-ar", "44100",
                "-ac", "2",
                "-t", str(audio_duration),
                output_path
            ]
            subprocess.run(cmd_concat, capture_output=True, text=True, check=True)

        return output_path

    @staticmethod
    def save_frame(i, combine_frame, img_output_path):
        # 保存图片
        output_path = f"{img_output_path}/{str(i).zfill(8)}.png"

        combine_frame = cv2.cvtColor(combine_frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, combine_frame)

        return output_path

    @staticmethod
    def write_video_ffmpeg(img_save_path, output_video, fps, audio_path, video_metadata):
        print(f"Writing image into video...")
        # 提取关键颜色信息
        pix_fmt = video_metadata.get("pix_fmt", "yuv420p")
        color_range = video_metadata.get("color_range", "1")
        color_space = video_metadata.get("color_space", "1")
        color_transfer = video_metadata.get("color_transfer", "1")
        color_primaries = video_metadata.get("color_primaries", "1")

        # 将图像序列转换为视频
        img_sequence_str = os.path.join(img_save_path, "%08d.png")  # 8位数字格式
        # 创建 FFmpeg 命令来合成视频
        cmd = [
            "ffmpeg",
            "-framerate", str(fps),  # 设置帧率
            "-i", img_sequence_str,  # 图像序列
            "-i", audio_path,  # 音频文件
            "-c:v", "libx264",  # 使用 x264 编码
            "-pix_fmt", pix_fmt,  # 设置像素格式
            "-color_range", color_range,  # 设置色彩范围
            "-colorspace", color_space,  # 设置色彩空间
            "-color_trc", color_transfer,  # 设置色彩传递特性
            "-color_primaries", color_primaries,  # 设置色彩基准
            "-c:a", "aac",  # 使用 AAC 编码音频
            "-b:a", "192k",  # 设置音频比特率
            "-ar", "44100",
            "-ac", "2",
            "-preset", "slow",  # 设置编码器预设
            "-crf", "18",  # 设置 CRF 值来控制视频质量
            "-y",
            output_video  # 输出文件路径
        ]

        # 执行 FFmpeg 命令
        subprocess.run(cmd, capture_output=True, text=True, check=True)

        return output_video
