
import onnxruntime as ort

MODEL_PATH = "yolov8n.onnx"

options = ort.SessionOptions()
options.intra_op_num_threads = 2
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

print("Providers:", session.get_providers())

for item in session.get_inputs():
    print(
        "INPUT:",
        item.name,
        "shape =", item.shape,
        "type =", item.type,
    )

for item in session.get_outputs():
    print(
        "OUTPUT:",
        item.name,
        "shape =", item.shape,
        "type =", item.type,
    )
