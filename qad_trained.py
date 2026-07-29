import math
import random
import time

# =====================================================================
# 1. PURE PYTHON METRICS EVALUATION
# =====================================================================

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


# =====================================================================
# 2. BASIC MATRIX OPERATIONS
# =====================================================================

def create_matrix(rows, cols, value=0.0):
    matrix = []
    for i in range(rows):
        row = []
        for j in range(cols):
            row.append(value)
        matrix.append(row)
    return matrix


def random_matrix(rows, cols):
    matrix = []
    for i in range(rows):
        row = []
        for j in range(cols):
            row.append(random.uniform(-1, 1))
        matrix.append(row)
    return matrix


def transpose(matrix):
    rows = len(matrix)
    cols = len(matrix[0])
    result = create_matrix(cols, rows)
    for i in range(rows):
        for j in range(cols):
            result[j][i] = matrix[i][j]
    return result


def matrix_multiply(A, B):
    rows = len(A)
    cols = len(B[0])
    common = len(B)
    result = create_matrix(rows, cols)

    for i in range(rows):
        for j in range(cols):
            total = 0
            for k in range(common):
                total += A[i][k] * B[k][j]
            result[i][j] = total

    return result


def add_bias(matrix, bias):
    rows = len(matrix)
    cols = len(matrix[0])
    result = create_matrix(rows, cols)

    for i in range(rows):
        for j in range(cols):
            result[i][j] = matrix[i][j] + bias[0][j]

    return result


# =====================================================================
# 3. FAKE QUANTIZATION
# =====================================================================

def quantize_dequantize(weights, bits):
    if bits == 32:  # 32 bit precision is full precision no quantization required
        return weights

    qmin = -(2 ** (bits - 1))     # -8 for 4-bit
    qmax = (2 ** (bits - 1)) - 1 #  7 for 4-bit

    maximum = 0

    for row in weights:  # Finds the largest weight magnitude
        for value in row:
            if abs(value) > maximum:
                maximum = abs(value)

    if maximum == 0:
        maximum = 0.000001

    scale = maximum / qmax
    new_weights = []

    for row in weights:
        new_row = []
        for value in row:
            q = round(value / scale)
            if q < qmin:
                q = qmin
            if q > qmax:
                q = qmax

            dq = q * scale
            new_row.append(dq)

        new_weights.append(new_row)

    return new_weights


# =====================================================================
# 4. SOFTMAX & QAD LOSS
# =====================================================================

def softmax(logits, temperature):
    result = []
    for row in logits:
        largest = row[0]
        for value in row:
            if value > largest:
                largest = value

        exp_values = []
        total = 0

        for value in row:
            # Numerically stable temperature-scaled softmax math
            e = math.exp((value - largest) / temperature)
            exp_values.append(e)
            total += e

        probs = []
        for e in exp_values:
            probs.append(e / total)

        result.append(probs)

    return result


def qad_loss(student_logits, teacher_logits, temperature):
    teacher_prob = softmax(teacher_logits, temperature)
    student_prob = softmax(student_logits, temperature)

    batch = len(student_logits)
    loss = 0
    gradient = create_matrix(batch, len(student_logits[0]))

    for i in range(batch):
        for j in range(len(student_logits[0])):
            pt = teacher_prob[i][j]
            ps = student_prob[i][j]

            loss += pt * math.log((pt + 1e-9) / (ps + 1e-9))  # KL Loss
            gradient[i][j] = (ps - pt) * temperature / batch

    loss = loss / batch
    loss = loss * temperature * temperature

    return loss, gradient


# =====================================================================
# 5. QUANTIZED LINEAR LAYER
# =====================================================================

class QuantizedLinearLayer:
    def __init__(self, input_size, output_size, bits):
        self.weight = random_matrix(input_size, output_size)
        self.bias = create_matrix(1, output_size)
        self.bits = bits
        self.input = None
        self.dw = None
        self.db = None

    def forward(self, x):
        self.input = x

        if self.bits == 32:
            w = self.weight
        else:
            w = quantize_dequantize(self.weight, self.bits)

        output = matrix_multiply(x, w)
        output = add_bias(output, self.bias)

        return output

    def backward(self, grad_output):
        input_T = transpose(self.input)
        self.dw = matrix_multiply(input_T, grad_output)
        self.db = create_matrix(1, len(grad_output[0]))

        for row in grad_output:
            for j in range(len(row)):
                self.db[0][j] += row[j]

    def update(self, lr):
        for i in range(len(self.weight)):
            for j in range(len(self.weight[0])):
                self.weight[i][j] -= lr * self.dw[i][j]

        for j in range(len(self.bias[0])):
            self.bias[0][j] -= lr * self.db[0][j]


# =====================================================================
# 6. TRAINING & EVALUATION
# =====================================================================

random.seed(0)

batch = 4
input_size = 5
output_size = 3

teacher = QuantizedLinearLayer(input_size, output_size, 32)
student = QuantizedLinearLayer(input_size, output_size, 4)

# Copy teacher weights to student
student.weight = []
for row in teacher.weight:
    student.weight.append(row[:])

student.bias = []
for row in teacher.bias:
    student.bias.append(row[:])

# Dummy Input
X = random_matrix(batch, input_size)

learning_rate = 0.1
temperature = 2

print("Training QAD...\n")

for epoch in range(10):
    teacher_logits = teacher.forward(X)
    student_logits = student.forward(X)

    loss, grad = qad_loss(student_logits, teacher_logits, temperature)

    student.backward(grad)
    student.update(learning_rate)

    print("Epoch", epoch + 1, "Loss =", round(loss, 6))


# =====================================================================
# 7. PERFORMANCE & METRICS EVALUATION
# =====================================================================

print("\n--- Evaluating Fine-Tuned Student Model ---")

# Measure Inference Speed
start_time = time.perf_counter()
final_student_logits = student.forward(X)
end_time = time.perf_counter()

total_inference_time = end_time - start_time
avg_latency_us = (total_inference_time / batch) * 1_000_000

final_teacher_logits = teacher.forward(X)

# Binarize outputs (> 0 -> Class 1, <= 0 -> Class 0)
y_true_binary = []
y_pred_binary = []

for i in range(len(final_teacher_logits)):
    for j in range(len(final_teacher_logits[0])):
        y_true_binary.append(1 if final_teacher_logits[i][j] > 0 else 0)
        y_pred_binary.append(1 if final_student_logits[i][j] > 0 else 0)

acc, prec, rec, f1, cm = compute_metrics(y_true_binary, y_pred_binary)

print("\n==================================================")
print("             QAD EVALUATION METRICS               ")
print("==================================================")
print(f"Total Predictions Assessed: {len(y_true_binary)}")
print("\n--- Classification Performance ---")
print(f"Accuracy : {acc:.4f} ({acc * 100:.2f}%)")
print(f"Precision: {prec:.4f} ({prec * 100:.2f}%)")
print(f"Recall   : {rec:.4f} ({rec * 100:.2f}%)")
print(f"F1-Score : {f1:.4f} ({f1 * 100:.2f}%)")

print("\nConfusion Matrix (TN FP / FN TP):")
print(f" [{cm[0][0]}  {cm[0][1]}]\n [{cm[1][0]}  {cm[1][1]}]")

print("\n--- Inference Time Performance ---")
print(f"Total Inference Time: {total_inference_time:.6f} seconds")
print(f"Avg Latency / Sample: {avg_latency_us:.3f} µs")
print("==================================================")
