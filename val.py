# YOLO-Trajectory validation script
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolo-trajectory.pt")  # path to trained weights
    model.val(
        data="ultralytics/cfg/datasets/dark-xj-seg.yaml",
        task="segment",
        split="val",
        imgsz=640,
        batch=16,
        project="runs/val",
        name="exp",
    )
