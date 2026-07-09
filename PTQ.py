import numpy as np

# CORE QUANTIZATION MATHEMATICS

def calc_scale_and_zp(min_val, max_val, num_bits=8):
    # Calculates step size (Scale) and origin alignment (Zero-Point)
    qmin, qmax = 0, (2**num_bits) - 1     #qmin and qmax are the min and max integer values that can be represented using a given number of bits - 2^8 = 256 (0-255)

    
    # Enforce that 0.0 is perfectly representable in the scale grid
    min_val = min(min_val, 0.0)   # we use this to make sure that floating point value 0.0 lies within the quantization range
    # if the data range is [2.45,9.0] and then we find the min_val = min(2.5,0.0) then the minimum is 0
    max_val = max(max_val, 0.0)
    
    if min_val == max_val:
        return 1.0, 0   #this means return scale as 1 nad return zero point as 0
        # to prevent the division by zero error
    # okay os why scale as 1  because max_val = 5 and min_val = 5 then 
    #scale = (max - min)/qmax-qmin = 5-5/255-0 = 0/255 = 0
    # q = (x/s) + z => x/0   creates divison by zero erro
    # z = qmin - min_val/scale
        
    scale = (max_val - min_val) / (qmax - qmin)
    initial_zp = qmin - (min_val / scale)   # 0 - (2/1) = -2 so negative values are not allowed in unit 8 so that's why we take the neutral value as 0
    zero_point = int(np.clip(np.round(initial_zp), qmin, qmax))   # np.clip generates a values that stays within the range
    
    return scale, zero_point

def quantize(tensor, scale, zero_point, num_bits=8):
    #Converts continuous 64-bit float numbers into discrete 8-bit integers
    qmin, qmax = 0, (2**num_bits) - 1  #2^num_bits = 2^8 = 256
    q_tensor = np.round(tensor / scale) + zero_point  # tensor is the input function   q = (x/s) + z
    return np.clip(q_tensor, qmin, qmax).astype(np.uint8) 
 
  # we quantize because we want to reduce the memory

def dequantize(q_tensor, scale, zero_point):  
    # Restores integer matrices back into working float approximations
    return scale * (q_tensor.astype(np.float32) - zero_point)   # x^ = scale(x-z)
# we dequantize because we want to perform calculcations uisng values that approxiamte the original floating point number


# 2. INT8 LINEAR LAYER STRUCTURE
 scale * (q_tensor.astype(np.float32) - zero_point)

    def forward(self, X):  # forward performs infernece/ prediction and X represent the current activation matrix
        if not self.is_quantized: 
            # --- CALIBRATION PHASE ---
            # Track the distribution boundaries of your activation data streams
            self.act_min = min(self.act_min, np.min(X))   
            # calcautes the minimum activation vlaue

            self.act_max = max(self.act_max, np.max(X))
            # calcautes the max activation vlaue
            

            return np.dot(X, self.W.T) + self.B   # y = xw +b   np.dot() is used ot perform the dot function between the vecotrs
            # np.dot will perform the dot function between vector
        else:
            # --- INT8 INFERENCE IN ACTION (Fake Quantization Math) ---
            # 1. Quantize & Dequantize incoming activation input
            q_X = quantize(X, self.act_scale, self.act_zp)   #floating point activations are converted to integer
            dq_X = dequantize(q_X, self.act_scale, self.act_zp)   #back from integer to floating point
            
            # 2. Quantize & Dequantize internal static layer weights
            q_W = quantize(self.W, self.w_scale, self.w_zp)   # same quanitze the weights floating-point weights are converted into integer 
            dq_W = dequantize(q_W, self.w_scale, self.w_zp)   # same quanitze the weights floating-point weights are converted into integer 
            
            # 3. Complete structural linear network matrix math
            return np.dot(dq_X, dq_W.T) + self.B

    def finalize_ptq(self):  # this functions is called after predictions has completed 
        # Processes collected data statistics to lock in INT8 constants
        self.w_scale, self.w_zp = calc_scale_and_zp(np.min(self.W), np.max(self.W))     # this calcultes the min and max weights  and then find min  and store them to the helper function
        self.act_scale, self.act_zp = calc_scale_and_zp(self.act_min, self.act_max)    # those collected stats to freeze the final quantization parameters for the activations.
        self.is_quantized = True


# 3. PIPELINE EXECUTION   #(pipeline are the basically steps or sequnce of processing stages where the output of the one stage become the input of hte another stage)

if __name__ == "__main__":   # this behaves like main entry point where we start calling functions
    np.random.seed(42)
    
    # Generate mock tracking data (Imagine this is your text or image dataset batches)
    # 100 samples total, each sample contains 4 numeric features 
    dataset = np.random.normal(loc=2.0, scale=1.5, size=(100, 4))   # loc is the center of the distribution(mean) = 2 so all values must be around 2
    
    calibration_data = dataset[:80]  # First 80 samples to map ranges    
    evaluation_data = dataset[80:]   # Last 20 samples to verify accuracy  # (for testing)

    # took a mock pre-trained network layer (Outputs 3 nodes, takes 4 inputs)
    mock_weights = np.random.uniform(-3.0, 3.0, size=(3, 4))   
    mock_bias = np.random.uniform(-0.5, 0.5, size=(3,))
    
    # Create the model layer instances
    fp32_layer = PTQLinearLayer(mock_weights, mock_bias)   # this will store the float32 and bias  this is the original data 
    ptq_layer = PTQLinearLayer(mock_weights, mock_bias)  # this ptq_layer we are goign to change 
    
    # the we will compare both these lines
    
    #  Run calibration data to learn activation statistics
    for batch in calibration_data:   #take on sample at a time from the calibration_data
        _ = ptq_layer.forward(batch.reshape(1, -1))    # forward fucntion pass the input to the model this
        # reshape will convet the 1d neural netowork into hte format expceted by the model  to 1 row and -1 column means(python calculate)
        
    #  Lock down quantization scales based on data profile
    ptq_layer.finalize_ptq()       #
    
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