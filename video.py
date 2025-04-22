import cv2
from ultralytics import YOLO
import torch
print(torch.cuda.is_available())  # 返回True表示PyTorch可以访问CUDA

# 加载模型
model = YOLO("models/yolo11s.pt", verbose=False)  # 或者使用 yolov8s.pt, yolov8m.pt 等

# 打开视频文件
cap = cv2.VideoCapture("media/cars.mp4")

# 获取视频的宽度和高度
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 创建视频写入对象
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('media/output.mp4', fourcc, 30.0, (width, height))

# 循环处理视频的每一帧
while cap.isOpened():
    # 读取一帧
    ret, frame = cap.read()
    if not ret:
        break

    # 进行目标检测
    results = model.predict(frame, verbose=False)  # 添加verbose=False参数

    # 解析结果并绘制边界框和标签
    for result in results:
        for box in result.boxes:
            if box.conf[0].item() >= 0.5:  # 只绘制置信度大于等于 0.5 的边界框
                x1, y1, x2, y2 = box.xyxy.tolist()[0]
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
                # 获取标签和置信度
                label = result.names[box.cls[0].item()]
                confidence = box.conf[0].item()
                
                # 绘制标签
                label_text = f"{label} {confidence:.2f}"
                cv2.putText(frame, label_text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 将处理后的帧写入输出视频文件
    out.write(frame)

    # 显示当前帧（可选）
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()
