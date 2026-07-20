import math

# 1. 4-bit NormalFloat (NF4) Quantization Functions


# NF4 uses 16 specific numbers optimized for standard normal distributions.


# This acts as our lookup table.
NF4_VALUES = [  # 16 normalized values using NF4 datatype 2^4 = 16  these values are placed near to zero using gaussian distribution and then after quantization each normalized weights are replaced by closest value
    -1.0, -0.6961928, -0.5250105, -0.3949175,  
    -0.2844414, -0.1847734, -0.0910500,  0.0,
     0.0795803,  0.1609302,  0.2461123,  0.3379152,
     0.4407098,  0.5626170,  0.7229568,  1.0
]

def quantize_to_nf4(value, scale):   # this function take weights normalize it nad find the closes value from 16 predefined nf4 values
    #Normalizes the value by scale, and finds the closest NF4 index (0 to 15)
    normalized = value / scale   # suppose value = 0.18 and scale = 0.2 = 0.9
    
    # Find the index of the NF4 value closest to our normalized value
    closest_index = 0
    min_distance = float('inf')  #minimum distance as infinity
    
    for i, nf4_val in enumerate(NF4_VALUES):
        distance = abs(normalized - nf4_val)  #how far the current NF4 value is from the normalized weight
        if distance < min_distance:
            min_distance = distance
            closest_index = i
            
    return closest_index




def dequantize_from_nf4(index, scale):
    # Looks up the NF4 float value using the index and scales it back up.
    return NF4_VALUES[index] * scale


# 2. Setup & Hyperparameters
learning_rate = 0.1
epochs = 5
alpha = 2.0  # LoRA scaling hyperparameter
rank = 1     # Low rank dimension (using 1 for simplicity)

# --- The "Pre-trained" Base Model Weight ---
# Imagine this is a parameter from a massive 70B LLM. We want to freeze it.
base_weight = 1.85   # this represent the pretrained weight

# Step A: Calculate absolute scale factor for this weight block
base_scale = abs(base_weight) if abs(base_weight) > 0 else 1e-5 #base scale is the (normalize the weight before quantization)
 
# Step B: Quantize it down to a 4-bit index (0-15) and FREEZE IT
frozen_q_index = quantize_to_nf4(base_weight, base_scale)  #  convert the pretrained weight into 4bit NF4 index
del base_weight # Delete the variable to simulate memory savings

# --- The Low-Rank Adapters (LoRA) ---
# Instead of fine-tuning the base weight, we train these two small adapters.
# W_change = (lora_B * lora_A) * (alpha / rank)
lora_A = 0.45  
lora_B = 0.00  # Initializing B to 0 ensures the adapter starts by doing nothing

# Scaling constant applied to LoRA pathway
lora_scaling = alpha / rank     

# Training dataset (x: input, y: target output)
training_data = [
    {"x": 0.5, "y": 1.0},
    {"x": 1.2, "y": 2.4},
    {"x": -0.8, "y": -1.6}
]

print(f"Base Weight safely frozen as 4-bit Index: {frozen_q_index}")
print(f"Initial LoRA Weights -> lora_A: {lora_A}, lora_B: {lora_B}\n")


# 3. Fine-Tuning Pathway (QLoRA Forward & Backward Pass)
for epoch in range(1, epochs + 1):
    total_loss = 0

    for sample in training_data:
        x = sample["x"]
        y_target = sample["y"]

        # --- Forward Pass ---
        
        # 1. On-the-fly Dequantization: temporarily wake up the base weight into FP32
        w_base_fp32 = dequantize_from_nf4(frozen_q_index, base_scale)
        
        # 2. Compute Base Model Prediction
        y_pred_base = x * w_base_fp32
        
        # 3. Compute LoRA Adapter Prediction 
        # Mathematically representing: x * lora_A * lora_B * scaling
        y_pred_lora = x * lora_A * lora_B * lora_scaling
        
        # 4. Final Combined Prediction
        y_pred = y_pred_base + y_pred_lora

        # Loss Calculation (MSE)
        loss = (y_pred - y_target) ** 2
        total_loss += loss

        # --- Backward Pass ---
        
        # Derivative of Loss with respect to Prediction
        d_pred = 2 * (y_pred - y_target)

        # Gradients flow ONLY into the LoRA parameters. 
        # The base model weight receives NO gradients because it's frozen!
        d_lora_B = d_pred * (x * lora_A * lora_scaling)
        d_lora_A = d_pred * (x * lora_B * lora_scaling)

        # Gradient Descent Updates
        lora_B = lora_B - learning_rate * d_lora_B
        lora_A = lora_A - learning_rate * d_lora_A

    print(f"Epoch: {epoch}")
    print(f"  Total Loss:   {round(total_loss, 6)}")
    print(f"  LoRA Weights: lora_A={round(lora_A, 4)}, lora_B={round(lora_B, 4)}")
    
    # Calculate what the total effective weight looks like now
    current_w_base = dequantize_from_nf4(frozen_q_index, base_scale)
    current_w_lora = lora_A * lora_B * lora_scaling
    print(f"  Total Combined Effective Weight: {round(current_w_base + current_w_lora, 4)}\n")