# Fake Quantization Function

def fake_quantize(value, min_val, max_val, bits=8):

    # Integer range for signed INT8
    qmin = -(2 ** (bits - 1))
    qmax = (2 ** (bits - 1)) - 1

    # Calculate scale
    scale = (max_val - min_val) / (qmax - qmin)

    # Prevent division by zero
    if scale == 0:
        scale = 0.00000001

    # Calculate zero point
    zero_point = round(qmin - (min_val / scale))

    #  zero point
    if zero_point < qmin:
        zero_point = qmin
    elif zero_point > qmax:
        zero_point = qmax

    # Quantize
    q_value = round(value / scale) + zero_point

    #  quantized value
    if q_value < qmin:
        q_value = qmin
    elif q_value > qmax:
        q_value = qmax

    # Dequantize
    dq_value = (q_value - zero_point) * scale

    return dq_value


# Hyperparameters

learning_rate = 0.1  # learning rate is used to check how much to change the weight after each training 
epochs = 5  # one complete pass through the entire training dataset 


# Initial weight
weight = 0.55

# range
BOUND_MIN = -2.0   # because the mostly floating point range lies near 0 and to make sure the values must between -2 and 2 
BOUND_MAX = 2.0

# Training dataset
training_data = [
    {"x": 0.5, "y": 1.0},   # x is the  input nad y is the correct output
    {"x": 1.2, "y": 2.4},
    {"x": -0.8, "y": -1.6}
]

print("Initial Weight:", weight)
print()


# Quantization-Aware Training


for epoch in range(1, epochs + 1):     # this epoch is the hyperparameter one complete step suppose epoch=5 range(1,6) 
    # excluded last item runs for 5 times 1 2 3 4 5

    total_loss = 0

    for sample in training_data:  # start from one sample 

        x = sample["x"]
        y_target = sample["y"]

        # Fake Quantization->  We fake quantize x and weight so that the model experiences the same precision loss it will experience after deployment on INT8 hardware.
        q_x = fake_quantize(x, BOUND_MIN, BOUND_MAX)      # The model learns how to work with quantized inputs.
        q_weight = fake_quantize(weight, BOUND_MIN, BOUND_MAX)   #The model learns how to work with quantized weights

        # Forward Pass
        y_pred = q_x * q_weight   #pred = x*w

        # Loss
        loss = (y_pred - y_target) * (y_pred - y_target)  
        total_loss += loss
        
        
        # Backward Pass

        # dLoss/dPrediction
        d_pred = 2 * (y_pred - y_target)  # thsi measures the error

    
        d_weight = d_pred * q_x     # how much weight cause that erorr

        # Gradient Descent
        weight = weight - learning_rate * d_weight  # update the weight so that the next prediction must be near to the range

    print("Epoch:", epoch)
    print("Total Loss:", round(total_loss, 6))
    print("Master Weight:", round(weight, 6))
    print()

# -----------------------------
# Final Deployment Weight
# -----------------------------
final_weight = fake_quantize(weight, BOUND_MIN, BOUND_MAX)

print("Training Complete")
print("Final Quantized Weight:", round(final_weight, 6))