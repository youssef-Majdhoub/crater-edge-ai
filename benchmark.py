
import time
import statistics

import cv2
import numpy as np
import onnxruntime as ort

from edge_detector import preprocess, postprocess

MODEL_PATH = "yolov8n.onnx"
IMAGE_PATH = "test.jpg"
CPU_THREADS = 2
WARMUP_RUNS = 10
MEASURED_RUNS = 100

options = ort.SessionOptions()
options.intra_op_num_threads = CPU_THREADS
options.inter_op_num_threads = 1
options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
options.graph_optimization_level = (
    ort.GraphOptimizationLevel.ORT_ENABLE_ALL
)

session = ort.InferenceSession(
    MODEL_PATH,
    sess_options=options,
    providers=["CPUExecutionProvider"],
)

input_name = session.get_inputs()[0].name

image = cv2.imread(IMAGE_PATH)
if image is None:
    raise FileNotFoundError(IMAGE_PATH)

tensor, transform = preprocess(image)

# Warm up the inference engine.
for _ in range(WARMUP_RUNS):
    session.run(None, {input_name: tensor})

inference_times = []
pipeline_times = []

for _ in range(MEASURED_RUNS):
    start = time.perf_counter()

    output = session.run(None, {input_name: tensor})
    inference_end = time.perf_counter()

    detections = postprocess(output, transform)
    pipeline_end = time.perf_counter()

    inference_times.append(inference_end - start)
    pipeline_times.append(pipeline_end - start)

mean_inference = statistics.mean(inference_times)
mean_pipeline = statistics.mean(pipeline_times)

print("\n--- BENCHMARK RESULTS ---")
print(f"CPU threads: {CPU_THREADS}")
print(f"Warm-up runs: {WARMUP_RUNS}")
print(f"Measured runs: {MEASURED_RUNS}")
print(f"Mean inference latency: {mean_inference * 1000:.2f} ms")
print(f"Inference FPS: {1 / mean_inference:.2f}")
print(f"Mean inference + NMS latency: {mean_pipeline * 1000:.2f} ms")
print(f"Inference + NMS FPS: {1 / mean_pipeline:.2f}")
print(f"Last detection count: {len(detections)}")
