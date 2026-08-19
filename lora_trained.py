import random
import time


# QUANTIZATION & MATRIX UTILITIES


def fake_quantize(value, min_val=-2.0, max_val=2.0, bits=8):
    
    qmin = -(2 ** (bits - 1))   # range from 0 to 255
    qmax = (2 ** (bits - 1)) - 1
    scale = (max_val - min_val) / (qmax - qmin) if max_val != min_val else 1e-8
    zero_point = round(qmin - (min_val / scale))
    zero_point = max(qmin, min(qmax, zero_point))

    q_value = round(value / scale) + zero_point
    q_value = max(qmin, min(qmax, q_value))

    return (q_value - zero_point) * scale

def quantize_matrix(matrix, min_val=-2.0, max_val=2.0):
    rows = len(matrix)
    cols = len(matrix[0])
    result = create_zero_matrix(rows, cols)
    for i in range(rows):
        for j in range(cols):
            result[i][j] = fake_quantize(matrix[i][j], min_val, max_val)
    return result

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
    result = create_zero_matrix(rows, cols)
    for i in range(rows):
        for j in range(cols):
            result[i][j] = A[i][j] + B[i][j]
    return result

def scalar_multiply(matrix, scalar):
    rows, cols = len(matrix), len(matrix[0])
    result = create_zero_matrix(rows, cols)
    for i in range(rows):
        for j in range(cols):
            result[i][j] = matrix[i][j] * scalar
    return result

def matrix_subtract(A, B):
    rows, cols = len(A), len(A[0])
    result = create_zero_matrix(rows, cols)
    for i in range(rows):
        for j in range(cols):
            result[i][j] = A[i][j] - B[i][j]
    return result


# LORA LAYER


class LORALinear:
    def __init__(self, in_features, out_features, rank=2, alpha=4):
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.W0 = gaussian_matrix(out_features, in_features, 0.5)
        self.A = gaussian_matrix(rank, in_features, 0.1)
        self.B = create_zero_matrix(out_features, rank)

        self.last_x = None
        self.last_lora = None
        self.grad_A = None
        self.grad_B = None

    def forward(self, x):
        self.last_x = x
        base = matrix_multiply(x, transpose(self.W0))
        self.last_lora = matrix_multiply(x, transpose(self.A))
        lora = matrix_multiply(self.last_lora, transpose(self.B))
        lora = scalar_multiply(lora, self.scaling)
        return matrix_add(base, lora)

    def backward(self, grad_output):
        scaled = scalar_multiply(grad_output, self.scaling)
        self.grad_B = matrix_multiply(transpose(scaled), self.last_lora)
        grad_intermediate = matrix_multiply(scaled, self.B)
        self.grad_A = matrix_multiply(transpose(grad_intermediate), self.last_x)

    def update(self, lr):
        for i in range(len(self.A)):
            for j in range(len(self.A[0])):
                self.A[i][j] -= lr * self.grad_A[i][j]

        for i in range(len(self.B)):
            for j in range(len(self.B[0])):
                self.B[i][j] -= lr * self.grad_B[i][j]

    def merged_weights(self):
        BA = matrix_multiply(self.B, self.A)
        BA = scalar_multiply(BA, self.scaling)
        return matrix_add(self.W0, BA)


# METRICS COMPUTATION (CLASSIFICATION)


def evaluate_performance(weights, X_data, Y_data):
    tp = fp = tn = fn = 0
    
    # Warmup
    _ = matrix_multiply(X_data, transpose(weights))

    start_time = time.perf_counter()
    
    # Forward Pass through Merged Weights
    predictions = matrix_multiply(X_data, transpose(weights))
    
    end_time = time.perf_counter()
    inference_time = end_time - start_time

    # Binarize outputs (> 0 -> 1, <= 0 -> 0)
    for i in range(len(Y_data)):
        for j in range(len(Y_data[0])):
            actual = 1 if Y_data[i][j] > 0 else 0
            pred = 1 if predictions[i][j] > 0 else 0

            if actual == 1 and pred == 1:
                tp += 1
            elif actual == 0 and pred == 1:
                fp += 1
            elif actual == 0 and pred == 0:
                tn += 1
            elif actual == 1 and pred == 0:
                fn += 1

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    return accuracy, precision, recall, inference_time



# DATASET & LORA TRAINING


random.seed(42)

# Test / Evaluation Dataset (Larger batch for stable metrics)
num_samples = 500
X = gaussian_matrix(num_samples, 8)
Y = gaussian_matrix(num_samples, 3)

layer = LORALinear(in_features=8, out_features=3, rank=2, alpha=4)
learning_rate = 0.01

# Train LoRA
for epoch in range(15):
    prediction = layer.forward(X)
    error = matrix_subtract(prediction, Y)

    total = 0
    rows, cols = len(error), len(error[0])

    for i in range(rows):
        for j in range(cols):
            total += error[i][j] * error[i][j]

    loss = total / (rows * cols)

    grad = create_zero_matrix(rows, cols)
    for i in range(rows):
        for j in range(cols):
            grad[i][j] = error[i][j] / rows

    layer.backward(grad)
    layer.update(learning_rate)



# EVALUATION (FP32 vs INT8)


# 1. FP32 Merged Weights Evaluation
fp32_weights = layer.merged_weights()
fp32_acc, fp32_prec, fp32_rec, fp32_time = evaluate_performance(fp32_weights, X, Y)

# 2. INT8 Quantized Merged Weights Evaluation
q_inputs = quantize_matrix(X)
q_weights = quantize_matrix(fp32_weights)
int8_acc, int8_prec, int8_rec, int8_time = evaluate_performance(q_weights, q_inputs, Y)

# Output Results
print("--- FP32 Evaluation Metrics ---")
print(f"Accuracy : {fp32_acc:.4f}")
print(f"Precision: {fp32_prec:.4f}")
print(f"Recall   : {fp32_rec:.4f}")
print(f"Inference Time: {fp32_time:.4f} seconds\n")

print("--- Quantized (INT8) Evaluation Metrics ---")
print(f"Accuracy : {int8_acc:.4f}")
print(f"Precision: {int8_prec:.4f}")
print(f"Recall   : {int8_rec:.4f}")
print(f"Inference Time: {int8_time:.4f} seconds")