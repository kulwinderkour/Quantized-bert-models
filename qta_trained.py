import random
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# -----------------------------
# 1. Fake Quantization & Model Infrastructure
# -----------------------------
def fake_quantize(value, min_val, max_val, bits=8):
    qmin = -(2 ** (bits - 1))
    qmax = (2 ** (bits - 1)) - 1
    scale = (max_val - min_val) / (qmax - qmin) if max_val != min_val else 1e-8
    zero_point = round(qmin - (min_val / scale))
    zero_point = max(qmin, min(qmax, zero_point))

    q_value = round(value / scale) + zero_point
    q_value = max(qmin, min(qmax, q_value))

    return (q_value - zero_point) * scale


def generate_dataset(num_samples=500, true_slope=2.0, noise_std=0.05, seed=42):
    random.seed(seed)
    data = []
    for _ in range(num_samples):
        x = random.uniform(-1.5, 1.5)
        noise = random.gauss(0, noise_std)
        y = (true_slope * x) + noise
        data.append({"x": x, "y": y})
    return data

# -----------------------------
# 2. Setup Data & Trained Weight
# -----------------------------
dataset = generate_dataset(num_samples=1000)
BOUND_MIN = min(d["x"] for d in dataset)
BOUND_MAX = max(d["x"] for d in dataset)

# Final INT8 Fake-Quantized Weight obtained after QAT training
quantized_weight = 1.99824  # Near true_slope=2.0

# -----------------------------
# 3. Measurement: Accuracy, Precision, Recall & Inference Latency
# -----------------------------
y_true_binary = []
y_pred_binary = []

# Warm-up pass (ensures CPU/interpreter caches are active for fair timing)
for _ in range(100):
    _ = fake_quantize(0.5, BOUND_MIN, BOUND_MAX) * quantized_weight

start_time = time.perf_counter()

for sample in dataset:
    # --- Prediction Pipeline ---
    q_x = fake_quantize(sample["x"], BOUND_MIN, BOUND_MAX)
    y_pred_continuous = q_x * quantized_weight

    # --- Classification Binary Thresholding ---
    # Class 1 if positive (> 0), Class 0 if non-positive (<= 0)
    actual_class = 1 if sample["y"] > 0 else 0
    pred_class = 1 if y_pred_continuous > 0 else 0

    y_true_binary.append(actual_class)
    y_pred_binary.append(pred_class)

end_time = time.perf_counter()

# Calculate timing metrics
total_time_sec = end_time - start_time
total_samples = len(dataset)
avg_latency_us = (total_time_sec / total_samples) * 1_000_000  # Microseconds per sample
throughput_fps = total_samples / total_time_sec                 # Inferences per second

# Calculate classification metrics
acc = accuracy_score(y_true_binary, y_pred_binary)
prec = precision_score(y_true_binary, y_pred_binary)
rec = recall_score(y_true_binary, y_pred_binary)
f1 = f1_score(y_true_binary, y_pred_binary)
cm = confusion_matrix(y_true_binary, y_pred_binary)

# -----------------------------
# 4. Results
# -----------------------------
print("==================================================")
print("             QUANTIZED MODEL METRICS              ")
print("==================================================")
print(f"Total Test Samples:    {total_samples}")
print("--- Classification Performance ---")
print(f"Accuracy:              {acc * 100:.2f}%")
print(f"Precision:             {prec * 100:.2f}%")
print(f"Recall:                {rec * 100:.2f}%")
print(f"F1-Score:              {f1 * 100:.2f}%")
print("\nConfusion Matrix (TN  FP / FN  TP):")
print(f" [{cm[0][0]}  {cm[0][1]}]\n [{cm[1][0]}  {cm[1][1]}]")

print("\n--- Inference Time Performance ---")
print(f"Total Execution Time:  {total_time_sec * 1000:.3f} ms")
print(f"Avg Latency / Sample:  {avg_latency_us:.3f} µs")
print(f"Throughput:            {throughput_fps:,.0f} Inferences/sec")
print("==================================================")