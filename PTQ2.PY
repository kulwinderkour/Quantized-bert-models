# CORE QUANTIZATION MATHEMATICS

def calc_scale_and_zp(min_val, max_val, num_bits=8):
    qmin = 0
    qmax = (2 ** num_bits) - 1

    # Ensure 0 lies inside the quantization range
    if min_val > 0:
        min_val = 0.0

    if max_val < 0:
        max_val = 0.0

    # Avoid division by zero
    if min_val == max_val:
        return 1.0, 0

    scale = (max_val - min_val) / (qmax - qmin)

    initial_zp = qmin - (min_val / scale)

    zero_point = round(initial_zp)

    # Clip zero point
    if zero_point < qmin:
        zero_point = qmin
    elif zero_point > qmax:
        
        zero_point = qmax

    return scale, zero_point


# ----------------------------
# Quantization
# ----------------------------

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

    return q_tensor


# ----------------------------
# Dequantization
# ----------------------------

def dequantize(q_tensor, scale, zero_point):
    output = []

    for value in q_tensor:
        output.append(scale * (value - zero_point))

    return output