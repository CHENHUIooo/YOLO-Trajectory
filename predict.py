# YOLO-Trajectory inference script
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolo-trajectory.pt")  # path to trained weights
    model.predict(
        source="path/to/images",  # image / directory / video
        task="segment",
        imgsz=640,
        conf=0.25,
        save=True,
        project="runs/predict",
        name="exp",
    )
