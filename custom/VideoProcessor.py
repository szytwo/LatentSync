import os
import subprocess

import moviepy.video.fx.all as vfx
from fastapi import UploadFile
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips

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
    def convert_video_to_25fps(video_path, video_metadata):
        """ 使用 MoviePy 将视频转换为 25 FPS """
        # 检查视频帧率
        r_frame_rate = video_metadata.get("r_frame_rate", "25/1")
        original_fps = eval(r_frame_rate.strip())  # 将字符串帧率转换为浮点数
        target_fps = 25

        if original_fps != target_fps:
            logging.info(f"视频帧率为 {original_fps}，转换为 25 FPS")
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
            logging.info("视频帧率已经是 25 FPS，无需转换")
            return video_path, original_fps

    @staticmethod
    def get_video_metadata(video_path):
        cmd = [
            "ffprobe", "-i", video_path, "-show_streams", "-select_streams", "v", "-hide_banner", "-loglevel", "error"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        metadata = {}
        for line in result.stdout.splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                metadata[key.strip()] = value.strip()
        logging.info(metadata)
        return metadata

    @staticmethod
    def process_video_with_audio(video_path: str, audio_path: str):
        output_path = add_suffix_to_filename(video_path, "_with")
        # 加载视频和音频
        video_clip = VideoFileClip(video_path)
        audio_clip = AudioFileClip(audio_path)

        target_duration = audio_clip.duration
        video_duration = video_clip.duration

        logging.info(f"音频时长: {target_duration}s, 视频时长: {video_duration}s")

        if video_duration < target_duration:
            # 当视频时长小于音频时，构造一个交替正序、倒序的视频序列
            clips = []
            current_duration = 0
            forward = True  # 标记当前是否正序播放
            while current_duration < target_duration:
                # 如果需要倒序，则应用 time_mirror 效果
                clip = video_clip if forward else video_clip.fx(vfx.time_mirror)
                clips.append(clip)
                current_duration += video_duration
                forward = not forward
            # 拼接所有片段，并截取到目标时长
            final_clip = concatenate_videoclips(clips).subclip(0, target_duration)
        elif video_duration > target_duration:
            # 当视频时长大于音频时，直接截取视频前 target_duration 秒
            final_clip = video_clip.subclip(0, target_duration)
        else:
            # 时长一致时，直接使用原视频
            final_clip = video_clip
        # 将音频设置到视频上
        final_clip = final_clip.set_audio(audio_clip)
        # 输出处理后的视频
        final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")

        return output_path
