import cv2
import numpy as np
import time

def start_ingestion(video_path):
    # 1. Initialize the software-emulated visual feed
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Error: Could not open lunar visual feed.")

    # Target constraints dictated by lightweight AI models (e.g., YOLO-nano)
    TARGET_WIDTH = 640
    TARGET_HEIGHT = 640
    
    # Pre-allocate a fixed memory buffer (1 Batch, 3 Channels, Height, Width)
    # This prevents the CPU from reallocating RAM every frame, preserving FPS
    input_tensor = np.empty((1, 3, TARGET_HEIGHT, TARGET_WIDTH), dtype=np.float32)

    print("Initiating automated continuous data flow...")

# 1. Define strict target dimensions
    TARGET_WIDTH = 640
    TARGET_HEIGHT = 640
    
    # 2. Pre-allocate RAM for the intermediate OpenCV processing steps
    resized_buffer = np.empty((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8)
    rgb_buffer = np.empty((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8)
    
    # 3. Pre-allocate the final memory block for the ONNX AI engine (Float32, NCHW layout)
    ai_input_tensor = np.empty((1, 3, TARGET_HEIGHT, TARGET_WIDTH), dtype=np.float32)

    print("Memory pre-allocated. Initiating zero-allocation loop...")

    while True:
        start_time = time.time()
        
        ret, frame = cap.read()
        if not ret:
            break
            
        # STEP A: Resize directly into the pre-allocated resized_buffer
        cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT), dst=resized_buffer, interpolation=cv2.INTER_LINEAR)
        
        # STEP B: Convert colors directly into the pre-allocated rgb_buffer
        cv2.cvtColor(resized_buffer, cv2.COLOR_BGR2RGB, dst=rgb_buffer)
        
        # STEP C: Normalize (0-255 to 0.0-1.0) and rearrange layout to CHW
        # We process this and write it directly into the final AI tensor buffer
        normalized_chw = np.transpose(rgb_buffer.astype(np.float32) / 255.0, (2, 0, 1))
        np.copyto(ai_input_tensor[0], normalized_chw)
        
        # --- THE DATA HANDOFF ---
        # The AI now reads from ai_input_tensor without any memory fragmentation.
        
        # Display the buffer to prove it works
        cv2.imshow("Edge Ingestion Feed", rgb_buffer)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        fps = 1.0 / (time.time() - start_time)
        print(f"Optimized Ingestion Speed: {fps:.1f} FPS")

    cap.release()

if __name__ == "__main__":
    start_ingestion("1011.mp4")