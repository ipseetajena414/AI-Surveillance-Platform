from ultralytics import YOLO

def main():
    print("Loading YOLOv8n model...")
    model = YOLO('yolov8n.pt')

    print("Downloading the COCO8 dataset (a small subset of COCO for testing)...")
    # By running a 1-epoch training on coco8.yaml, ultralytics will automatically
    # download the dataset to your active directory or ultralytics datasets directory.
    # We use coco8 instead of the full coco because the full coco is >19GB.
    model.train(data='coco8.yaml', epochs=1, imgsz=640)
    
    print("Dataset download complete!")

if __name__ == "__main__":
    main()
