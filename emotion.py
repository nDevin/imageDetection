import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm
import os
from datetime import datetime
from ultralytics import YOLO
import ffmpeg
import shutil

class EmotionDetector:
    def __init__(self):
        # 初始化情绪识别模型
        self.model_name = "dima806/facial_emotions_image_detection"
        print("正在加载情绪识别模型...")
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForImageClassification.from_pretrained(self.model_name)
        self.model.eval()
        
        # 加载YOLO人脸检测模型
        print("正在加载YOLO人脸检测模型...")
        self.face_detector = YOLO('models/yolov8s-face.pt')
        print("所有模型加载完成!")
        
        # 颜色映射：不同情绪用不同颜色显示
        self.color_map = {
            'sad': (255, 0, 0),      # 蓝色
            'disgust': (0, 0, 255),  # 红色
            'angry': (0, 0, 128),    # 深红色
            'neutral': (128, 128, 128), # 灰色
            'fear': (128, 0, 128),   # 紫色
            'surprise': (0, 255, 255), # 黄色
            'happy': (0, 255, 0)     # 绿色
        }

    def predict_emotion(self, image):
        """预测单张图片的情绪"""
        inputs = self.processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][predicted_class].item()
            
        return self.model.config.id2label[predicted_class], confidence

    def process_frame(self, frame):
        """使用YOLO处理视频帧"""
        # YOLO人脸检测
        results = self.face_detector(frame, conf=0.5)  # 可以调整置信度阈值
        emotions_in_frame = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # 获取边界框坐标
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                
                # 提取人脸区域，稍微扩大检测框以包含更多上下文
                h = y2 - y1
                w = x2 - x1
                y1_extended = max(0, y1 - int(0.1 * h))
                y2_extended = min(frame.shape[0], y2 + int(0.1 * h))
                x1_extended = max(0, x1 - int(0.1 * w))
                x2_extended = min(frame.shape[1], x2 + int(0.1 * w))
                
                face = frame[y1_extended:y2_extended, x1_extended:x2_extended]
                if face.size == 0:  # 检查裁剪区域是否有效
                    continue
                    
                face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
                
                # 预测情绪
                emotion, emotion_conf = self.predict_emotion(face_pil)
                emotions_in_frame.append(emotion)
                
                # 获取对应的颜色
                color = self.color_map.get(emotion, (0, 255, 0))
                
                # 绘制人脸框
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # 添加标签
                label = f"{emotion}: {emotion_conf:.2f} (Face: {conf:.2f})"
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                
                # 绘制半透明背景
                overlay = frame.copy()
                cv2.rectangle(overlay, (x1, y1 - 30), (x1 + text_size[0] + 10, y1), color, -1)
                cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
                
                # 绘制文本
                cv2.putText(frame, label, (x1 + 5, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame, emotions_in_frame

    def process_video(self, video_path, output_path=None, save_stats=True):
        """处理视频文件，去除最后3秒，保留音频"""
        # 创建临时文件用于存储无音频的处理后视频
        temp_video_path = f'temp_{datetime.now().strftime("%Y%m%d_%H%M%S")}.mp4'
        
        # 读取原始视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("错误：无法打开视频文件")
            return
        
        # 获取视频信息
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 计算要处理的帧数（去除最后3秒）
        frames_to_process = max(0, total_frames - 3 * fps)
        
        # 创建输出视频文件
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f'output_video_{timestamp}.mp4'
        
        # 使用x264编码器
        fourcc = cv2.VideoWriter_fourcc(*'x264')
        out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
        
        # 统计变量
        emotion_stats = {emotion: 0 for emotion in self.color_map.keys()}
        frame_count = 0
        
        # 处理进度条
        pbar = tqdm(total=frames_to_process, desc="处理视频帧")
        
        while cap.isOpened() and frame_count < frames_to_process:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 处理帧
            processed_frame, emotions = self.process_frame(frame)
            
            # 更新统计信息
            for emotion in emotions:
                emotion_stats[emotion] += 1
            frame_count += 1
            
            # 写入输出视频
            out.write(processed_frame)
            pbar.update(1)
        
        pbar.close()
        cap.release()
        out.release()
        
        # 使用ffmpeg合并视频和音频
        print("\n正在处理音频...")
        try:
            # 获取处理后视频的时长
            probe = ffmpeg.probe(temp_video_path)
            processed_duration = float(probe['streams'][0]['duration'])
            
            # 合并视频和音频
            input_video = ffmpeg.input(temp_video_path)
            input_audio = ffmpeg.input(video_path).audio.filter('atrim', duration=processed_duration)
            
            # 输出最终视频
            stream = ffmpeg.output(input_video, input_audio, output_path, 
                                 vcodec='copy', acodec='aac')
            
            # 运行ffmpeg命令
            print("正在合成最终视频...")
            ffmpeg.run(stream, overwrite_output=True, quiet=True)
            
            print("音频处理完成")
            
        except Exception as e:
            print(f"处理音频时出错: {str(e)}")
            # 如果音频处理失败，至少保留处理后的无声视频
            if os.path.exists(temp_video_path):
                shutil.move(temp_video_path, output_path)
        
        # 删除临时文件
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        
        # 保存统计信息
        if save_stats:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            stats_file = f'emotion_stats_{timestamp}.txt'
            with open(stats_file, 'w', encoding='utf-8') as f:
                f.write("情绪统计报告\n")
                f.write("=" * 30 + "\n")
                f.write(f"视频总帧数: {frame_count}\n")
                f.write("各情绪出现次数:\n")
                for emotion, count in emotion_stats.items():
                    percentage = (count / frame_count) * 100 if frame_count > 0 else 0
                    f.write(f"{emotion}: {count} 次 ({percentage:.2f}%)\n")
        
        print(f"\n处理完成！")
        print(f"输出视频保存为: {output_path}")
        if save_stats:
            print(f"统计报告保存为: {stats_file}")

    def process_image(self, image_path):
        """处理单张图片"""
        img = cv2.imread(image_path)
        if img is None:
            print("错误：无法加载图片")
            return None, []
        
        return self.process_frame(img)

def main():
    detector = EmotionDetector()
    
    print("\n选择模式:")
    print("1: 摄像头实时检测")
    print("2: 图片检测")
    print("3: 视频文件处理")
    
    mode = input("\n请输入模式编号 (1/2/3): ")
    
    if mode == "1":
        cap = cv2.VideoCapture(0)
        print("\n摄像头已启动，按 'q' 退出")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            processed_frame, _ = detector.process_frame(frame)
            cv2.imshow('Emotion Detection', processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        
    elif mode == "2":
        image_path = input("\n请输入图片路径: ")
        processed_frame, emotions = detector.process_image(image_path)
        if processed_frame is not None:
            cv2.imshow('Emotion Detection', processed_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
    elif mode == "3":
        video_path = input("\n请输入视频文件路径: ")
        save_output = input("是否保存处理后的视频? (y/n): ").lower() == 'y'
        save_stats = input("是否生成情绪统计报告? (y/n): ").lower() == 'y'
        
        if save_output:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f'output_video_{timestamp}.mp4'
        else:
            output_path = None
            
        detector.process_video(video_path, output_path, save_stats)

if __name__ == "__main__":
    main()