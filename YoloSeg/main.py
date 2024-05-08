from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n-seg.pt')  # load a pretrained model (recommended for training)

# Train the model
model.train(data="datasets/captcha2024.yaml", task="segment", mode="train", workers=0, batch=4, epochs=300, device=0)
