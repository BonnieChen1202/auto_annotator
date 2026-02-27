from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from ultralytics import YOLO
from PIL import Image
import io

# 1. 初始化 FastAPI 应用
app = FastAPI(title="AI 自动标注 API")

# 新增这一段来解决跨域问题，允许网页访问接口
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 允许所有前端网页访问
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. 初始化 YOLO 模型
# 建议在程序启动时就加载模型，这样不用每次收到请求都重新加载，速度更快。
# 第一次运行会自动下载 yolov8n.pt
model = YOLO('yolov8n.pt')


# 3. 编写 POST 接口
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 步骤 A：读取前端上传的图片文件
    image_bytes = await file.read()
    # 将字节流转换为 PIL 图像格式，YOLO 可以直接处理 PIL 图像
    image = Image.open(io.BytesIO(image_bytes))

    # 步骤 B：使用 YOLO 进行推理
    results = model(image)

    # 步骤 C：提取检测结果
    detections = []
    # results[0] 包含单张图片的所有预测结果
    for box in results[0].boxes:
        # 1. 提取坐标 xyxy 并转换为普通的 Python 列表
        xyxy = box.xyxy[0].tolist()

        # 2. 提取类别 ID 和置信度 (需要从 tensor 转换为普通数值)
        class_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())

        # 3. 获取 YOLO 模型自带的类别名称字典，拿到对应的字符串名字
        class_name = model.names[class_id]

        # 将这个框的信息塞进列表
        detections.append({
            "class_id": class_id,
            "class_name": class_name,
            "confidence": round(conf, 2),  # 保留两位小数
            # 坐标通常需要整数像素值，方便前端画图
            "bbox": [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
        })

    # 步骤 D：组装并返回 JSON 格式
    return {
        "code": 200,
        "message": "success",
        "data": {
            "image_size": image.size,  # 返回原图尺寸 (width, height)
            "detections": detections
        }
    }


# 4. 启动服务器
if __name__ == "__main__":
    # reload=True 表示如果你修改了代码，服务器会自动重启
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)