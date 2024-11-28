import cv2
from ultralytics import YOLO

# 加载模型
model = YOLO("./models/yolo11s.pt")  # 假设我们使用的是 YOLOv8 的 nano 模型

# 读取图片
image_path = "media/image.png"
image = cv2.imread(image_path)

# 进行目标检测
results = model.predict(source=image, save=True, save_txt=True)  # 保存检测结果为图片和文本文件

# 打印检测结果
for result in results:
    boxes = result.boxes
    for box in boxes:
        if box.conf >= 0.5:  # 只绘制概率在 0.5 及以上的检测结果
            print(f"类别: {box.cls}, 置信度: {box.conf}, 边界框: {box.xyxy}")

            # 提取类别标签和置信度
            label = f"{model.names[int(box.cls)]} {float(box.conf):.2f}"
            
            # 绘制边界框和类别标签
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# 显示图像
cv2.imshow("Image with Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
