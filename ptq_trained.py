#this trained model contain the libraries of the scikit learn
import sys
import platform
# Bypass broken WMI lookup in Python 3.14 on Windows
platform.machine = lambda: "AMD64" if sys.maxsize > 2**31 - 1 else "x86"

import math
import time
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

# ----------------------------------------
# 1. CORE QUANTIZATION UTILITIES
# ----------------------------------------


# this will find the min and max range of the int 
def find_min_max(data):
    minimum = min(data)
    maximum = max(data)
    return minimum, maximum
# return the maximum and minimum
def calc_scale_and_zp(min_val, max_val, num_bits=8):
    # calcuate the scale and zero point 
    qmin = 0
    qmax = (2 ** num_bits) - 1

    if min_val > 0:
        min_val = 0.0
    if max_val < 0:
        max_val = 0.0

    if min_val == max_val:
        return 1.0, 0   # return the scale as 1 to avoid the zero divison erorr 

    scale = (max_val - min_val) / (qmax - qmin)
    initial_zp = qmin - (min_val / scale)
    zero_point = round(initial_zp)   # zero point formula using round of initial zero point

    if zero_point < qmin:
        zero_point = qmin
    elif zero_point > qmax:
        zero_point = qmax

    return scale, zero_point

def quantize(tensor, scale, zero_point, num_bits=8):
    qmin = 0
    qmax = (2 ** num_bits) - 1
    q_tensor = []
    for value in tensor:
        q = round(value / scale) + zero_point
        if q < qmin:
            q = qmin
        elif q > qmax:
            q = qmax
        q_tensor.append(int(q))
    return q_tensor   #this will return the quantized value

def dequantize(q_tensor, scale, zero_point):
    return [scale * (val - zero_point) for val in q_tensor]

def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-max(min(z, 500), -500)))

# ----------------------------------------
# 2. LINEAR LAYER WITH TRAINING & PTQ
# ----------------------------------------

class QuantizedLinearClassifier:
    def __init__(self, input_dim):
        # Initialize weights and bias randomly
        self.W = np.random.randn(input_dim) * 0.01
        self.B = 0.0

        self.is_quantized = False
        self.act_min = float('inf')
        self.act_max = float('-inf')

        self.w_scale = 0.0
        self.w_zp = 0
        self.act_scale = 0.0
        self.act_zp = 0

    def forward_single(self, x_vector):
        if not self.is_quantized:
            # Track activation range for PTQ calibration
            min_val, max_val = find_min_max(x_vector)
            if min_val < self.act_min:
                self.act_min = min_val
            if max_val > self.act_max:
                self.act_max = max_val

            # Compute floating point dot product
            z = sum(x * w for x, w in zip(x_vector, self.W)) + self.B
            return sigmoid(z)
        else:
            # Quantize & dequantize activation and weights
            q_X = quantize(x_vector, self.act_scale, self.act_zp)
            dq_X = dequantize(q_X, self.act_scale, self.act_zp)

            q_W = quantize(self.W, self.w_scale, self.w_zp)
            dq_W = dequantize(q_W, self.w_scale, self.w_zp)

            z = sum(x * w for x, w in zip(dq_X, dq_W)) + self.B
            return sigmoid(z)

    def fit(self, X_train, y_train, epochs=10, lr=0.1):
        """Train floating-point weights using Logistic Regression (Gradient Descent)"""
        start_time = time.perf_counter()
        
        for epoch in range(epochs):
            for x_vec, y_true in zip(X_train, y_train):
                # Forward pass FP32
                z = sum(x * w for x, w in zip(x_vec, self.W)) + self.B
                pred = sigmoid(z)
                
                # Gradients
                error = pred - y_true
                for j in range(len(self.W)):
                    self.W[j] -= lr * error * x_vec[j]
                self.B -= lr * error

        training_time = time.perf_counter() - start_time
        return training_time

    def finalize_ptq(self, num_bits=8):
        weight_min, weight_max = find_min_max(self.W)
        self.w_scale, self.w_zp = calc_scale_and_zp(weight_min, weight_max, num_bits)
        self.act_scale, self.act_zp = calc_scale_and_zp(self.act_min, self.act_max, num_bits)
        self.is_quantized = True

    def predict(self, X):
        start_time = time.perf_counter()
        preds = []
        for x_vec in X:
            prob = self.forward_single(x_vec)
            preds.append(1 if prob >= 0.5 else 0)
        elapsed_time = time.perf_counter() - start_time
        return preds, elapsed_time

# ----------------------------------------
# 3. DATA PREPROCESSING & RUN SCRIPT
# ----------------------------------------

# 1. Load Data
df = pd.read_csv('imdb_movies.csv')
df = df.dropna(subset=['overview', 'score'])

# 2. Define binary target variable (e.g., High Score >= 70)
df['label'] = (df['score'] >= 70.0).astype(int)

# 3. Vectorize Text Overview using TF-IDF
vectorizer = TfidfVectorizer(max_features=100) # Keep small feature space for raw python execution speed
X_sparse = vectorizer.fit_transform(df['overview'])
X = X_sparse.toarray().tolist()
y = df['label'].values.tolist()

# 4. Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Initialize Model & Train FP32 Model
model = QuantizedLinearClassifier(input_dim=len(X[0]))

print("--- Training FP32 Model ---")
train_time = model.fit(X_train, y_train, epochs=5, lr=0.1)
print(f"Training Time: {train_time:.4f} seconds")

# 6. Evaluate FP32 Model
fp32_preds, fp32_inf_time = model.predict(X_test)
print("\n--- FP32 Evaluation Metrics ---")
print(f"Accuracy : {accuracy_score(y_test, fp32_preds):.4f}")
print(f"Precision: {precision_score(y_test, fp32_preds, zero_division=0):.4f}")
print(f"Recall   : {recall_score(y_test, fp32_preds, zero_division=0):.4f}")
print(f"Inference Time: {fp32_inf_time:.4f} seconds")

# 7. Finalize PTQ (Quantize Weights and Activation Scale)
model.finalize_ptq(num_bits=8)

# 8. Evaluate Quantized INT8 Model
int8_preds, int8_inf_time = model.predict(X_test)
print("\n--- Quantized (INT8) Evaluation Metrics ---")
print(f"Accuracy : {accuracy_score(y_test, int8_preds):.4f}")
print(f"Precision: {precision_score(y_test, int8_preds, zero_division=0):.4f}")
print(f"Recall   : {recall_score(y_test, int8_preds, zero_division=0):.4f}")
print(f"Inference Time: {int8_inf_time:.4f} seconds")