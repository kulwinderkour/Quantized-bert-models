# Example pretrained weights
weights = [
    0.15,
    -0.82,
    1.25,
    -2.10,
    0.73,
    1.84,
    -1.42,
    0.05
]
# SAMPLE WEIGHTS 

num_bits = 8   #this means 2^8 = 256(0 to 255) map every real number in this range 

def find_min_max(data):  # defines the floating point range
    minimum = data[0]
    maximum = data[0]

    for value in data:
        if value < minimum:
            minimum = value
        if value > maximum:
            maximum = value

    return minimum, maximum   # this is used for the quality of quantization whihc is entirely depend on true range 


# CORE QUANTIZATION MATHEMATICS
    
def calc_scale_and_zp(min_val, max_val, num_bits):   # min_val = minimum and max_val = maximum 
    #scales are the small steps
    qmin = 0
    qmax = (2 ** num_bits) - 1  # range will be from 0 to 255 and this 1 integer represent how many floating point 

    # Ensure 0 lies inside the quantization range
    if min_val > 0:
        min_val = 0.0

    if max_val < 0:
        max_val = 0.0

    # Avoid division by zero
    if min_val == max_val:
        return 1.0, 0  

    scale = (max_val - min_val) / (qmax - qmin)   #scale = (5-5)/(255-0) = 0  so make sure return scale as 1

    initial_zp = qmin - (min_val / scale)   # initial_zp = 0-(5/0)  ->leads to division by zero error

    zero_point = round(initial_zp)

    # Clip zero point
    if zero_point < qmin:  
        zero_point = qmin
    elif zero_point > qmax:
        
        zero_point = qmax

    return scale, zero_point



# Quantization


def quantize(tensor, scale, zero_point, num_bits=8):
    qmin = 0
    qmax = (2 ** num_bits) - 1   #should be from 0 to 255

    q_tensor = []

    for value in tensor:
        q = round(value / scale) + zero_point  

        if q < qmin:   # make sure to keep the quantized values inside the range 0 to 255 
            q = qmin  #and q must be greater than 0 and less than 255
        elif q > qmax:
            q = qmax

        q_tensor.append(int(q))

    return q_tensor


# Dequantization


def dequantize(q_tensor, scale, zero_point):    # convert the integer back to floating points
    output = []

    for value in q_tensor:
        output.append(scale * (value - zero_point))
    #dq_x = s*(value-zeropoint)
    return output


# Main Program


minimum, maximum = find_min_max(weights)

scale, zero_point = calc_scale_and_zp(
    minimum,
    maximum,
    num_bits
)

quantized_weights = quantize(
    weights,
    scale,
    zero_point,
    num_bits
)

dequantized_weights = dequantize(
    quantized_weights,
    scale,
    zero_point
)

# ----------------------------------------
# Display Results
# ----------------------------------------

print("Original Weights")
print(weights)

print("\nMinimum =", minimum)
print("Maximum =", maximum)

print("\nScale =", scale)
print("Zero Point =", zero_point)

print("\nQuantized Weights")
print(quantized_weights)

print("\nDequantized Weights")
print(dequantized_weights)






# ----------------------------------------
# INT8 Linear Layer (PTQ)
# ----------------------------------------

class LinearLayer:

    def __init__(self, weights, bias):

        self.W = weights
        self.B = bias   # this allows model to shift

        self.is_quantized = False

        self.act_min = 999999   # smallest activation so far
        self.act_max = -999999  # largest 
 
        self.w_scale = 0  # scale for the weights   (conver the floating point value of weight intp integer)
        self.w_zp = 0 # zero point for the weight  (inside the range)

        self.act_scale = 0   # scale used to quanitze activations this will tell to conver the floating point values into integer use how much scale
        self.act_zp = 0   # to shift inside the range (0-255)


    # ----------------------------------------
    # Forward Pass
    # ----------------------------------------
 
    def forward(self, X):  ## X = [0.4, -0.8, 1.5]  min = -0.8 and max 1.5
       
        if not self.is_quantized:  # not of self.is_quantize means The model is not yet quantized, and not false = true then execute the block

            minimum, maximum = find_min_max(X)  # [-0.8, 1.5]

            if minimum < self.act_min:
                self.act_min = minimum  # self.act_min = -0.8
            # self.act_min will store hte smallest activation observed

            if maximum > self.act_max:
                self.act_max = maximum  # self.act_max = 1.5
            output = 0   # initalize the output varibale = 0

            for i in range(len(X)):   #loop through every element of X
                output += X[i] * self.W[i]   # output = XW + B

            output += self.B

            return output

        else:   # this execute when the model has already been quantiezed

            # Quantize input activations
            q_X = quantize(
                X,
                self.act_scale,  
                self.act_zp
            )

            dq_X = dequantize(
                q_X,
                self.act_scale,
                self.act_zp
            )

            # Quantize weights
            q_W = quantize(
                self.W,
                self.w_scale,
                self.w_zp
            )

            dq_W = dequantize(
                q_W,
                self.w_scale,
                self.w_zp
            )

            output = 0 

            for i in range(len(dq_X)):
                output += dq_X[i] * dq_W[i]

            output += self.B

            return output


    # ----------------------------------------
    # Finalize PTQ
    # ----------------------------------------

    def finalize_ptq(self):  #self.W = [0.15, -0.82, 1.25, -2.1, 0.73]

        weight_min, weight_max = find_min_max(self.W)    
        # this will find the smallest and largest weight (-2.1,1.25)

        self.w_scale, self.w_zp = calc_scale_and_zp(   
            weight_min,
            weight_max,
            num_bits
        )

        self.act_scale, self.act_zp = calc_scale_and_zp(   # use the activation scale and activation zp whihc is used to quantize the input activations
            self.act_min,
            self.act_max,
            num_bits
        )

        self.is_quantized = True   # this means that the model is ready to use quantiztion


# ----------------------------------------
# Example
# ----------------------------------------

layer = LinearLayer(
    weights=[0.4, -0.6, 0.8],
    bias=0.2
)

X = [1.5, -0.5, 2.0]

print("\nFloating Point Output")
print(layer.forward(X))

layer.finalize_ptq()

print("\nQuantized Output")
print(layer.forward(X))
