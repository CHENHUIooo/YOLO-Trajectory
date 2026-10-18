# YOLO-Trajectory training script
from ultralytics import YOLO

if __name__ == "__main__":
    # Paper model (DMSAB + CSP-MKFB + C3k2-PMSB); scale suffix can be n/s/m/l/x
    model = YOLO("ultralytics/cfg/models/11/yolo-trajectory.yaml")
    # model = YOLO("yolo-trajectory.pt")  # resume from a checkpoint

    model.train(
        data="ultralytics/cfg/datasets/dark-xj-seg.yaml",
        task="segment",
        imgsz=640,
        epochs=200,
        batch=16,
        cache=False,
        workers=8,
        device="0",
        optimizer="SGD",
        amp=True,
        close_mosaic=10,
        project="runs/train",
        name="exp",
    )
