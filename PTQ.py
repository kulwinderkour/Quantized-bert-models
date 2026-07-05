import numpy as np

# ==========================================
# 1. CORE QUANTIZATION MATHEMATICS
# ==========================================

def calc_scale_and_zp(min_val, max_val, num_bits=8):
    """Calculates step size (Scale) and origin alignment (Zero-Point)."""
    qmin, qmax = 0, (2**num_bits) - 1
    
    # Enforce that 0.0 is perfectly representable in the scale grid
    min_val = min(min_val, 0.0)
    max_val = max(max_val, 0.0)
    
    if min_val == max_val:
        return 1.0, 0
        
    scale = (max_val - min_val) / (qmax - qmin)
    initial_zp = qmin - (min_val / scale)
    zero_point = int(np.clip(np.round(initial_zp), qmin, qmax))
    
    return scale, zero_point

def quantize(tensor, scale, zero_point, num_bits=8):
    """Converts continuous 64-bit float numbers into discrete 8-bit integers."""
    qmin, qmax = 0, (2**num_bits) - 1
    q_tensor = np.round(tensor / scale) + zero_point
    return np.clip(q_tensor, qmin, qmax).astype(np.uint8)

def dequantize(q_tensor, scale, zero_point):
    """Restores integer matrices back into working float approximations."""
    return scale * (q_tensor.astype(np.float32) - zero_point)


# ==========================================
# 2. INT8 LINEAR LAYER STRUCTURE
# ==========================================

class PTQLinearLayer:
    def __init__(self, weights, bias):
        self.W = weights  # Matrix shape: (out_features, in_features)
        self.B = bias     # Matrix shape: (out_features,)
        
        # Calibration state flags and memory registers
        self.is_quantized = False
        self.act_min = float('inf')
        self.act_max = float('-inf')
        
        # Quantization variables
        self.w_scale, self.w_zp = None, None
        self.act_scale, self.act_zp = None, None

    def forward(self, X):
        if not self.is_quantized:
            # --- CALIBRATION PHASE ---
            # Track the distribution boundaries of your activation data streams
            self.act_min = min(self.act_min, np.min(X))
            self.act_max = max(self.act_max, np.max(X))
            return np.dot(X, self.W.T) + self.B
        else:
            # --- INT8 INFERENCE IN ACTION (Fake Quantization Math) ---
            # 1. Quantize & Dequantize incoming activation input
            q_X = quantize(X, self.act_scale, self.act_zp)
            dq_X = dequantize(q_X, self.act_scale, self.act_zp)
            
            # 2. Quantize & Dequantize internal static layer weights
            q_W = quantize(self.W, self.w_scale, self.w_zp)
            dq_W = dequantize(q_W, self.w_scale, self.w_zp)
            
            # 3. Complete structural linear network matrix math
            return np.dot(dq_X, dq_W.T) + self.B

    def finalize_ptq(self):
        """Processes collected data statistics to lock in INT8 constants."""
        self.w_scale, self.w_zp = calc_scale_and_zp(np.min(self.W), np.max(self.W))
        self.act_scale, self.act_zp = calc_scale_and_zp(self.act_min, self.act_max)
        self.is_quantized = True


# ==========================================
# 3. PIPELINE EXECUTION
# ==========================================
if __name__ == "__main__":
    np.random.seed(42)
    
    # Generate mock tracking data (Imagine this is your text or image dataset batches)
    # 100 samples total, each sample contains 4 numeric features
    dataset = np.random.normal(loc=2.0, scale=1.5, size=(100, 4))
    calibration_data = dataset[:80]  # First 80 samples to map ranges
    evaluation_data = dataset[80:]   # Last 20 samples to verify accuracy
    
    # Initialize a mock pre-trained network layer (Outputs 3 nodes, takes 4 inputs)
    mock_weights = np.random.uniform(-3.0, 3.0, size=(3, 4))
    mock_bias = np.random.uniform(-0.5, 0.5, size=(3,))
    
    # Create the model layer instances
    fp32_layer = PTQLinearLayer(mock_weights, mock_bias)
    ptq_layer = PTQLinearLayer(mock_weights, mock_bias)
    
    # STEP 1: Run calibration data to learn activation statistics
    for batch in calibration_data:
        _ = ptq_layer.forward(batch.reshape(1, -1))
        
    # STEP 2: Lock down quantization scales based on data profile
    ptq_layer.finalize_ptq()
    
    # STEP 3: Run evaluation verification comparing FP32 vs INT8 precision
    print("--- PTQ Parameter Configurations ---")
    print(f"Weight Quantization   -> Scale: {ptq_layer.w_scale:.5f} | Zero-Point: {ptq_layer.w_zp}")
    print(f"Activation Calibration -> Scale: {ptq_layer.act_scale:.5f} | Zero-Point: {ptq_layer.act_zp}\n")
    
    print("--- Processing Test Dataset Samples ---")
    total_error = 0.0
    
    for sample in evaluation_data:
        x_in = sample.reshape(1, -1)
        
        # Run raw float math pipeline
        out_fp32 = fp32_layer.forward(x_in)
        
        # Run quantized math pipeline
        out_int8 = ptq_layer.forward(x_in)
        
        # Track mathematical variance difference
        total_error += np.mean(np.abs(out_fp32 - out_int8))
        
    print(f"Mean Quantization Noise Discrepancy: {total_error / len(evaluation_data):.6f}")