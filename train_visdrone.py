from ultralytics import YOLO

def main():
    print("Initializing YOLOv8 Nano model...")
    # We start from the pretrained nano model because it is the fastest to train
    model = YOLO('yolov8n.pt')

    print("\n--- Starting VisDrone Training ---")
    print("NOTE: The VisDrone dataset is very large (~2.5GB download, ~10GB uncompressed).")
    print("Since this machine is using a CPU for training, we are limiting training to 3 epochs.")
    print("If you want a highly accurate production model, change 'epochs=3' to 'epochs=100'")
    print("and run this on a machine with an NVIDIA GPU.")
    print("----------------------------------\n")

    # Start training. Ultralytics will automatically download the dataset
    # specified by VisDrone.yaml if it is not already present.
    # We use a batch size of 4 to avoid overwhelming the system memory.
    results = model.train(
        data='VisDrone.yaml',
        epochs=3,          # Reduced to 3 epochs for CPU testing
        imgsz=640,         # Standard YOLOv8 image size
        batch=4,           # Small batch size for CPU stability
        device='cpu',      # Explicitly set to CPU
        project='runs/detect',
        name='visdrone_custom'
    )

    print("\nTraining complete!")
    print("Your new custom model weights are saved at: runs/detect/visdrone_custom/weights/best.pt")

if __name__ == "__main__":
    main()
