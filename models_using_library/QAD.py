import numpy as np

# 1. QUANTIZATION UTILITIES (STE & Symmetric Quantization)


def quantize_dequantize(x, num_bits=4):
    """
    Simulates fake quantization (Symmetric).
    Maps float to discrete integer bins, then scales it back to float.
    """
    if num_bits == 32:
        return x
        
    # Calculate the max bound for the given bits
    qmin = -(2 ** (num_bits - 1))
    qmax = (2 ** (num_bits - 1)) - 1
    
    # Calculate scale factor
    max_val = np.max(np.abs(x)) if np.max(np.abs(x)) > 0 else 1e-7
    scale = max_val / qmax
    
    # Quantize, clamp, and dequantize (Fake Quantization)
    q_x = np.round(x / scale)
    q_x = np.clip(q_x, qmin, qmax)
    dq_x = q_x * scale
    
    return dq_x

# =====================================================================
# 2. CUSTOM LAYER WITH BACKPROP & FAKE QUANTIZATION
# =====================================================================
class QuantizedLinearLayer:
    def __init__(self, in_features, out_features, num_bits=4):
        # Initialize full-precision weights
        self.weight = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        self.bias = np.zeros((1, out_features))
        #Bias is initialized to zero because there is no advantage to starting it with random values.
        self.num_bits = num_bits   #This stores the quantization precision inside the object.
        
        # Gradients and state caches
        self.dw = None
        self.db = None
        self.x = None

    def forward(self, x, eval_mode=False):   #eval_mode=False → An optional parameter. In this code it is not used, so it has no effect.
        self.x = x
        # If training or testing as a quantized student, apply fake quantization to weights
        if self.num_bits < 32:
            w_sim = quantize_dequantize(self.weight, self.num_bits)
            # Original weights ->FP32 Weights-> Fake Quantization ->Quantized Integers ->Dequantized Floats ->stored in w_sim
        else:
            w_sim = self.weight   #other wise no quantization
            
        return np.dot(x, w_sim) + self.bias     #this is the forward layer
    
    def backward(self, grad_output):
        # Using Straight-Through Estimator (STE): 
        # The gradient passes through the quantization step unaffected.
        self.dw = np.dot(self.x.T, grad_output)
        self.db = np.sum(grad_output, axis=0, keepdims=True)
        
        # Gradient to pass to previous layer if needed
        grad_input = np.dot(grad_output, self.weight.T)
        return grad_input

    def update_weights(self, lr):
        self.weight -= lr * self.dw
        self.bias -= lr * self.db

# =====================================================================
# 3. SOFTMAX, TEMPERATURE SCALING, & KL DIVERGENCE LOSS
# =====================================================================
def softmax(logits, temperature=1.0):
    """Computes softmax probabilities with temperature scaling."""
    scaled_logits = logits / temperature
    # Subtract max for numerical stability
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=-1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

def compute_qad_loss_and_grad(student_logits, teacher_logits, temperature=2.0):
    """
    Computes KL Divergence loss and the analytical gradient 
    with respect to the student logits.
    """
    batch_size = student_logits.shape[0]
    
    # Get soft targets from teacher and student
    p_teacher = softmax(teacher_logits, temperature)
    p_student = softmax(student_logits, temperature)
    
    # KL Divergence Loss
    kl_loss = np.sum(p_teacher * np.log((p_teacher + 1e-9) / (p_student + 1e-9))) / batch_size
    # Scale loss by T^2 as per standard distillation literature
    scaled_loss = kl_loss * (temperature ** 2)
    
    # Gradient of KL Divergence w.r.t student logits: dL/dz_s
    # Factoring in the Temperature scaling multiplier (T^2 * 1/T leaves a factor of T)
    grad_logits = (p_student - p_teacher) * temperature / batch_size
    
    return scaled_loss, grad_logits

# =====================================================================
# 4. RUNNING THE QAD TRAINING LOOP
# =====================================================================
if __name__ == "__main__":
    np.random.seed(42)    #this is the random state 42  as a convention
    
    # Hyperparameters
    batch_size = 4
    in_features = 8
    out_features = 3  # e.g., a 3-class classification problem
    learning_rate = 0.1
    temperature = 2.0
    epochs = 5
    
    # Generate mock input data X
    X = np.random.randn(batch_size, in_features)
    
    # 1. Create a Teacher Layer (Stays at Full FP32 Precision)
    teacher = QuantizedLinearLayer(in_features, out_features, num_bits=32)
    
    # 2. Create a Student Layer (To be compressed to 4-bit)
    student = QuantizedLinearLayer(in_features, out_features, num_bits=4)
    # Give them identical starting weights to clearly trace accuracy recovery
    student.weight = teacher.weight.copy()
    student.bias = teacher.bias.copy()    

    print("--- Initial Discrepancy (Due to 4-bit Quantization Error) ---")
    t_logits_init = teacher.forward(X)
    s_logits_init = student.forward(X)
    initial_loss, _ = compute_qad_loss_and_grad(s_logits_init, t_logits_init, temperature)
    print(f"Initial QAD (KL) Loss: {initial_loss:.6f}\n")

    print("--- Starting Quantization-Aware Distillation Loop ---")
    for epoch in range(1, epochs + 1):
        # Forward pass: Teacher (FP32)
        teacher_logits = teacher.forward(X)
        
        # Forward pass: Student (Simulated 4-bit)
        student_logits = student.forward(X)
        
        # Compute Knowledge Distillation Loss & Gradient
        loss, grad_wrt_logits = compute_qad_loss_and_grad(student_logits, teacher_logits, temperature)
        
        # Backward pass through student layer (STE passes gradient straight through)
        student.backward(grad_wrt_logits)
        
        # Update full-precision student weights to absorb/counteract quantization noise
        student.update_weights(learning_rate)
        
        print(f"Epoch {epoch}/{epochs} -> QAD Loss: {loss:.6f}")