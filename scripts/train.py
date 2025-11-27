from ultralytics import YOLO
import os

# paths and datasets info
DATA_YAML = "../dataset/data.yaml"
TEST_IMAGES = "../dataset/test/images"
RESULTS_DIR = "../runs/detect/vegetable_train"

# initiate yolov8 model
model = YOLO("yolov8n.pt")

# train the model
model.train(
    data=DATA_YAML,
    epochs=20,
    imgsz=640,
    batch=16,
    project=RESULTS_DIR,  # where results are saved
    name="exp2",           # folder name for this run
    exist_ok=True,         # overwrite if folder exists
)

# validate the model 
best_model_path = os.path.join(RESULTS_DIR, "exp1/weights/best.pt")
metrics = model.val(model=best_model_path, data=DATA_YAML)
print("Validation metrics:", metrics)

# test the model using test images
TEST_IMAGES = "../dataset/test/images"  # folder with test images
output_dir = os.path.join(RESULTS_DIR, "exp1/predictions")

results = model.predict(
    source=TEST_IMAGES,
    conf=0.25,       # confidence threshold
    save=True,       # save annotated images
    save_txt=True,   # save predictions in txt files
    project=output_dir,
    name="predictions",
    exist_ok=True
)

print("Inference done! Check predictions in:", output_dir)