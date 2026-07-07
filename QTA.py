import numpy
import random

# STE - straight through estimator used in QTA that let gradients to pass through the non differentaible functions like rounding and quantization


def fake_quantize(value, min_val, max_val, bits=8):   #this min_val and max_val defines the bits range
    qmin = -(2 ** (bits - 1))  #2^(bits-1)
    qmax = (2 ** (bits - 1)) - 1
    # qmin/qmax define the integer range (-128 to 127 for INT8).

    scale = (max_val - min_val) / (qmax - qmin)
    # scale maps floating-point values to integer levels
    if abs(scale) < 1e-8:
        scale = 1e-8     #extremely tiny number close to zero this will prevent the scale to be 0 and from division by zero also 

    # The safety check prevents division by zero. if max = min 
    zero_point = round(qmin - (min_val / scale))
    zero_point = max(qmin, min(qmax, zero_point))   # we use this function because sometimes zero_point goes beyond range (0-255) -> min(255,280) = 255 and max(0,255) so the final value will be 280 -> 255

    # zero_point determines which integer corresponds to real value 0.
    # q_value performs quantization (rounding to the nearest integer).
    q_value = round(value / scale) + zero_point
    q_value = max(qmin, min(qmax, q_value))  # this is again used clamping and to fix the values if they goes beyound range
    # Clamping ensures values stay inside the INT8 range.


    # dq_value converts the integer back to floating point.
    #dequantization
    dq_value = (q_value - zero_point) * scale
    return dq_value


# Hyperparameters
learning_rate = 0.1  #learning_rate controls update size.
epochs = 5  #epochs is the number of passes over the dataset.

# Let's initialize a single weight parameter
weight = 0.55

# Target system boundaries (pretend we calibrated these ranges ahead of time)
# Inputs and weights typically sit in a known boundary, say -2.0 to 2.0
BOUND_MIN, BOUND_MAX = -2.0, 2.0  #define the calibration range used to compute scale


# Training data: Input (x) and expected Target (y)
# Let's say the true underlying function is roughly: y = 2 * x
training_data = [
    {"x": 0.5, "y": 1.0},
    {"x": 1.2, "y": 2.4},
    {"x": -0.8, "y": -1.6}
]
# The desired relationship is approximately y = 2x. The model must learn a weight close to 2.


print(f"Initial Floating-Point Weight: {weight:.4f}\n")
print("--- Starting Quantization-Aware Training ---")

for epoch in range(1, epochs + 1):   # epoch is one full pass time in training 
    total_loss = 0
    
    for sample in training_data:   # loop through every single training daata 
        x = sample["x"]   # where x is the sample input
        y_target = sample["y"]  # y is the output we are expecting from the model
        
      
        # FORWARD PASS WITH FAKE QUANTIZATION
      
        # We pass BOTH the input and the weight through our low-res filter.
        # This injects the exact precision loss that happens in an INT8 chip.
        q_x = fake_quantize(x, BOUND_MIN, BOUND_MAX, bits=8)
        q_weight = fake_quantize(weight, BOUND_MIN, BOUND_MAX, bits=8)
        
        # Compute prediction using the pixelated/quantized components
        y_pred = q_x * q_weight
        
        # Calculate Squared Error Loss
        loss = (y_pred - y_target) ** 2
        total_loss += loss
        
      
        # BACKWARD PASS (The STE Secret)
      
        # Loss derivative relative to prediction: dLoss/dPred = 2 * (y_pred - y_target)
        d_pred = 2 * (y_pred - y_target)
        
        # Normal calculus rule for Pred = x * weight would be: d_weight = d_pred * x
        # BUT because we are doing QAT, we calculate gradients based on the 
        # QUANTIZED inputs, bypassing the rounding function's true derivative (0).
        # This is the Straight-Through Estimator in pure code action!
        d_weight = d_pred * q_x    #this is where backpropogation is happening and it calculates the gradient of the weight  
        
        # Update our master high-precision floating-point weight
        weight = weight - (learning_rate * d_weight)  # updating hte weight to reduce the loss
        
    print(f"Epoch {epoch} | Total Loss: {total_loss:.6f} | Master Weight Float: {weight:.4f}")

print("\n--- Training Complete ---")
final_quantized_weight = fake_quantize(weight, BOUND_MIN, BOUND_MAX, bits=8)  
print(f"Final Quantized Weight ready for deployment: {final_quantized_weight:.4f}")