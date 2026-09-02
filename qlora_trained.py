import random
import time


# 1. PURE PYTHON METRICS (NO SKLEARN DEPENDENCY)


def compute_metrics(y_true, y_pred):
    tp = fp = tn = fn = 0
    for actual, pred in zip(y_true, y_pred):
        if actual == 1 and pred == 1:
            tp += 1
        elif actual == 0 and pred == 1:
            fp += 1
        elif actual == 0 and pred == 0:
            tn += 1
        elif actual == 1 and pred == 0:
            fn += 1

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return accuracy, precision, recall, f1, [[tn, fp], [fn, tp]]



# 2. 4-BIT NORMALFLOAT (NF4) QUANTIZATION ENGINE


NF4_VALUES = [
    -1.0, -0.6961928, -0.5250105, -0.3949175,
    -0.2844414, -0.1847734, -0.0910500,  0.0,
     0.0795803,  0.1609302,  0.2461123,  0.3379152,
     0.4407098,  0.5626170,  0.7229568,  1.0
]

def quantize_val_to_nf4(value, scale):
    normalized = value / scale if scale != 0 else 0.0
    closest_index = 0
    min_dist = float('inf')
    for i, nf4_val in enumerate(NF4_VALUES):
        dist = abs(normalized - nf4_val)
        if dist < min_dist:
            min_dist = dist
            closest_index = i
    return closest_index

def quantize_matrix_to_nf4(matrix):
    rows, cols = len(matrix), len(matrix[0])
    q_matrix, scales = [], []
    for i in range(rows):
        max_val = max(abs(x) for x in matrix[i])
        scale = max_val if max_val > 0 else 1e-8
        scales.append(scale)
        q_row = [quantize_val_to_nf4(matrix[i][j], scale) for j in range(cols)]
        q_matrix.append(q_row)
    return q_matrix, scales

def dequantize_matrix_from_nf4(q_matrix, scales):
    rows, cols = len(q_matrix), len(q_matrix[0])
    fp32_matrix = []
    for i in range(rows):
        scale = scales[i]
        row = [NF4_VALUES[q_matrix[i][j]] * scale for j in range(cols)]
        fp32_matrix.append(row)
    return fp32_matrix



# 3. MATRIX OPERATION UTILITIES


def create_zero_matrix(rows, cols):
    return [[0.0 for _ in range(cols)] for _ in range(rows)]

def gaussian_matrix(rows, cols, std=0.1):
    return [[random.gauss(0, std) for _ in range(cols)] for _ in range(rows)]

def transpose(matrix):
    rows, cols = len(matrix), len(matrix[0])
    result = create_zero_matrix(cols, rows)
    for i in range(rows):
        for j in range(cols):
            result[j][i] = matrix[i][j]
    return result

def matrix_multiply(A, B):
    rows_A, cols_A = len(A), len(A[0])
    cols_B = len(B[0])
    result = create_zero_matrix(rows_A, cols_B)
    for i in range(rows_A):
        for j in range(cols_B):
            total = 0.0
            for k in range(cols_A):
                total += A[i][k] * B[k][j]
            result[i][j] = total
    return result

def matrix_add(A, B):
    rows, cols = len(A), len(A[0])
    return [[A[i][j] + B[i][j] for j in range(cols)] for i in range(rows)]

def matrix_subtract(A, B):
    rows, cols = len(A), len(A[0])
    return [[A[i][j] - B[i][j] for j in range(cols)] for i in range(rows)]

def scalar_multiply(matrix, scalar):
    rows, cols = len(matrix), len(matrix[0])
    return [[matrix[i][j] * scalar for j in range(cols)] for i in range(rows)]



# 4. QLORA LINEAR LAYER CLASS


class QLORALinear:
    def __init__(self, in_features, out_features, rank=2, alpha=4):
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        raw_W0 = gaussian_matrix(out_features, in_features, std=0.5)
        self.q_W0, self.W0_scales = quantize_matrix_to_nf4(raw_W0)
        del raw_W0

        self.A = gaussian_matrix(rank, in_features, std=0.1)
        self.B = create_zero_matrix(out_features, rank)

        self.last_x = None
        self.last_lora = None
        self.grad_A = None
        self.grad_B = None

    def forward(self, X):
        self.last_x = X
        W0_fp32 = dequantize_matrix_from_nf4(self.q_W0, self.W0_scales)
        base_out = matrix_multiply(X, transpose(W0_fp32))

        self.last_lora = matrix_multiply(X, transpose(self.A))
        lora_out = matrix_multiply(self.last_lora, transpose(self.B))
        lora_out = scalar_multiply(lora_out, self.scaling)

        return matrix_add(base_out, lora_out)

    def backward(self, grad_output):
        scaled_grad = scalar_multiply(grad_output, self.scaling)
        self.grad_B = matrix_multiply(transpose(scaled_grad), self.last_lora)
        grad_inter = matrix_multiply(scaled_grad, self.B)
        self.grad_A = matrix_multiply(transpose(grad_inter), self.last_x)

    def update(self, lr):
        for i in range(len(self.A)):
            for j in range(len(self.A[0])):
                self.A[i][j] -= lr * self.grad_A[i][j]

        for i in range(len(self.B)):
            for j in range(len(self.B[0])):
                self.B[i][j] -= lr * self.grad_B[i][j]

    def merged_weights(self):
        W0_fp32 = dequantize_matrix_from_nf4(self.q_W0, self.W0_scales)
        BA = matrix_multiply(self.B, self.A)
        BA = scalar_multiply(BA, self.scaling)
        return matrix_add(W0_fp32, BA)



# 5. TRAINING & EVALUATION PIPELINE


random.seed(42)

num_samples = 500
in_dim = 8
out_dim = 3

X_train = gaussian_matrix(num_samples, in_dim)
Y_train = gaussian_matrix(num_samples, out_dim)

qlora_layer = QLORALinear(in_features=in_dim, out_features=out_dim, rank=2, alpha=4)
learning_rate = 0.02
epochs = 10

print("--- Starting QLoRA Fine-Tuning ---")
print(f"Base Weight Matrix: {out_dim}x{in_dim} (Stored in 4-bit NF4)")
print(f"LoRA Adapters: A({qlora_layer.rank}x{in_dim}), B({out_dim}x{qlora_layer.rank}) in FP32\n")

for epoch in range(1, epochs + 1):
    preds = qlora_layer.forward(X_train)
    error = matrix_subtract(preds, Y_train)

    rows, cols = len(error), len(error[0])
    total_loss = sum(error[i][j] ** 2 for i in range(rows) for j in range(cols))
    avg_loss = total_loss / (rows * cols)

    grad = create_zero_matrix(rows, cols)
    for i in range(rows):
        for j in range(cols):
            grad[i][j] = error[i][j] / rows

    qlora_layer.backward(grad)
    qlora_layer.update(learning_rate)

    print(f"Epoch {epoch:2d}/{epochs} | Training Loss: {avg_loss:.6f}")



# 6. EVALUATION METRICS


print("\n--- Evaluating Fine-Tuned Model ---")

final_weights = qlora_layer.merged_weights()

# Warm-up pass
_ = matrix_multiply(X_train, transpose(final_weights))

start_time = time.perf_counter()
test_predictions = matrix_multiply(X_train, transpose(final_weights))
end_time = time.perf_counter()

total_inference_time = end_time - start_time
avg_latency_us = (total_inference_time / num_samples) * 1_000_000

y_true_binary = []
y_pred_binary = []

for i in range(len(Y_train)):
    for j in range(len(Y_train[0])):
        y_true_binary.append(1 if Y_train[i][j] > 0 else 0)
        y_pred_binary.append(1 if test_predictions[i][j] > 0 else 0)

acc, prec, rec, f1, cm = compute_metrics(y_true_binary, y_pred_binary)



print("             QLoRA EVALUATION METRICS             ")

print(f"Total Test Predictions: {len(y_true_binary)}")
print("\n--- Classification Performance ---")
print(f"Accuracy : {acc:.4f} ({acc * 100:.2f}%)")
print(f"Precision: {prec:.4f} ({prec * 100:.2f}%)")
print(f"Recall   : {rec:.4f} ({rec * 100:.2f}%)")
print(f"F1-Score : {f1:.4f} ({f1 * 100:.2f}%)")

print("\nConfusion Matrix (TN FP / FN TP):")
print(f" [{cm[0][0]}  {cm[0][1]}]\n [{cm[1][0]}  {cm[1][1]}]")

print("\n--- Inference Time Performance ---")
print(f"Total Inference Time: {total_inference_time:.4f} seconds")
print(f"Avg Latency / Sample: {avg_latency_us:.3f} µs")
