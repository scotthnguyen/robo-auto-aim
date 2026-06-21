import time
import numpy as np
import onnxruntime as ort

MODEL_PATH = "armor_detector/model/mlp.onnx"
WARMUP = 100
RUNS = 1000

sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
inp = sess.get_inputs()[0]
print(f"Input: {inp.name}  shape={inp.shape}  dtype={inp.type}")

# NHWC layout: batch=1, H=28, W=20, C=1
dummy = np.random.rand(1, 20, 28, 1).astype(np.float32)
feed = {inp.name: dummy}

# Warmup
for _ in range(WARMUP):
    sess.run(None, feed)

# Timed runs
latencies = []
for _ in range(RUNS):
    t0 = time.perf_counter()
    sess.run(None, feed)
    latencies.append((time.perf_counter() - t0) * 1000)

arr = np.array(latencies)
print(f"\nMLP inference over {RUNS} runs:")
print(f"  mean   : {arr.mean():.3f} ms")
print(f"  std    : {arr.std():.3f} ms")
print(f"  p50    : {np.percentile(arr, 50):.3f} ms")
print(f"  p99    : {np.percentile(arr, 99):.3f} ms")
print(f"  min/max: {arr.min():.3f} / {arr.max():.3f} ms")
print(f"\nMax theoretical throughput: {1000/arr.mean():.0f} classifications/sec")
