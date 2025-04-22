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

class FaceDetector:
    def __init__(self):
        # 加载YOLO人脸检测模型
        print("正在加载YOLO人脸检测模型...")
        self.face_detector = YOLO('models/yolov8s-face.pt')
        print("模型加载完成!")

    def process_frame(self, frame):
        """使用YOLO处理视频帧"""
        # YOLO人脸检测
        results = self.face_detector(frame, conf=0.5)  # 可以调整置信度阈值
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # 获取边界框坐标
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                
                # 绘制人脸框 (绿色)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 添加标签
                label = f"Face: {conf:.2f}"
                # 修复字体名称拼写错误
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                
                # 绘制半透明背景
                overlay = frame.copy()
                cv2.rectangle(overlay, (x1, y1 - 30), (x1 + text_size[0] + 10, y1), (0, 255, 0), -1)
                cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
                
                # 绘制文本
                # 修复putText方法参数顺序
                cv2.putText(frame, label, (x1 + 5, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame  # 现在只返回处理后的帧

    def process_video(self, video_path, output_path=None):
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
        if not out.isOpened():  # 检查 VideoWriter 是否成功打开
            print("错误：无法创建输出视频文件")
            cap.release()
            return

        # 初始化 frame_count 变量
        frame_count = 0
        
        # 处理进度条
        pbar = tqdm(total=frames_to_process, desc="处理视频帧")
        
        while cap.isOpened() and frame_count < frames_to_process:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 处理帧
            processed_frame = self.process_frame(frame)
            
            # 写入输出视频
            out.write(processed_frame)
            pbar.update(1)
            frame_count += 1  # 更新 frame_count
        
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
        
        print(f"\n处理完成！")
        print(f"输出视频保存为: {output_path}")

    def process_image(self, image_path):
        """处理单张图片"""
        img = cv2.imread(image_path)
        if img is None:
            print("错误：无法加载图片")
            return None
        return self.process_frame(img)

def main():
    detector = FaceDetector()
    
    print("\n选择模式:")
    print("1: 图片检测")
    print("2: 视频文件处理")
    
    mode = input("\n请输入模式编号 (1/2): ")
    
    scale_increment = 0.2
    window_name = 'Face Detection'

    if mode == "1":
        # 模式2: 图片检测
        image_path = input("\n请输入图片路径: ").strip('"\'')
        original_img = cv2.imread(image_path)
        if original_img is None:
            print("错误：无法加载图片")
        else:
            try:
                # 预先处理完整图片（只执行一次人脸检测）
                processed_img = detector.process_image(image_path)
                if processed_img is None:
                    return
                
                # 使用处理后的图片作为显示基准
                display_img = processed_img.copy()
                h, w = display_img.shape[:2]
                
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                # 初始化状态变量
                crop_x, crop_y = 0, 0  # 修正变量名，定义 crop_y
                dragging = False
                start_x = start_y = 0
                need_update = True
                img_scale = 1.0  # 使用新变量避免全局变量冲突

                # 修改鼠标回调函数中的变量引用
                def mouse_callback(event, x, y, flags, _):  
                    # 移除无法nonlocal的h,w变量
                    nonlocal img_scale, need_update, crop_x, crop_y, dragging, start_x, start_y
                    if event == cv2.EVENT_MOUSEWHEEL:
                        old_scale = img_scale  # 改为使用局部缩放因子
                        if flags > 0:
                            img_scale = min(img_scale + scale_increment, 6.0)
                        else:
                            img_scale = max(img_scale - scale_increment, 0.5)
                        # 更新计算使用img_scale
                        new_w = int(w * img_scale)
                        new_h = int(h * img_scale)
                        x_img = (crop_x + x) / old_scale
                        y_img = (crop_y + y) / old_scale
                        crop_x = int(x_img * img_scale - x)
                        crop_y = int(y_img * img_scale - y)
                        crop_x = max(0, min(crop_x, new_w - w))
                        crop_y = max(0, min(crop_y, new_h - h))
                        need_update = True
                    elif event == cv2.EVENT_LBUTTONDOWN:
                        dragging = True
                        start_x, start_y = x, y
                    elif event == cv2.EVENT_MOUSEMOVE and dragging:
                        delta_x = x - start_x
                        delta_y = y - start_y
                        new_w = int(w * img_scale)
                        new_h = int(h * img_scale)
                        crop_x -= delta_x
                        crop_y -= delta_y
                        crop_x = max(0, min(crop_x, new_w - w))
                        crop_y = max(0, min(crop_y, new_h - h))
                        start_x, start_y = x, y
                        need_update = True
                    elif event == cv2.EVENT_LBUTTONUP:
                        dragging = False

                cv2.setMouseCallback(window_name, mouse_callback)

                # 修改显示循环部分：
                # 删除缓存字典
                # scaled_img_cache = {} 
                
                while True:
                    if need_update:
                        # 直接实时缩放，不缓存
                        new_w = int(w * img_scale)
                        new_h = int(h * img_scale)
                        scaled_img = cv2.resize(display_img, (new_w, new_h))
                        
                        # 计算裁剪区域
                        y1 = max(0, min(crop_y, new_h - h))
                        x1 = max(0, min(crop_x, new_w - w))
                        y2 = y1 + h
                        x2 = x1 + w
                        
                        # 确保裁剪区域不越界
                        try:
                            img = scaled_img[y1:y2, x1:x2]
                        except Exception as e:
                            print(f"图像处理异常: {str(e)}")
                            # 自动重置显示参数
                            img_scale = 1.0
                            crop_x = crop_y = 0
                            img = display_img
                        
                        cv2.imshow(window_name, img)
                        need_update = False
                    
                    key = cv2.waitKey(10)
                    # 检查窗口是否被关闭
                    if key != -1 or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                        break

                cv2.destroyAllWindows()
            except cv2.error as e:
                print(f"OpenCV 操作出错: {e}")
        
    elif mode == "2":
        video_path = input("\n请输入视频文件路径: ")
        save_output = input("是否保存处理后的视频? (y/n): ").lower() == 'y'
        
        if save_output:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f'output_video_{timestamp}.mp4'
        else:
            output_path = None
            
        detector.process_video(video_path, output_path)

if __name__ == "__main__":
    main()