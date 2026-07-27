import math
import random


# 1. BASIC MATRIX OPERATIONS


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
    common = len(B)  #common = len(B) tells the program how many pairs of numbers should be multiplied and added together to calculate one element of the result matrix

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



# 2. FAKE QUANTIZATION


def quantize_dequantize(weights, bits):

    if bits == 32:   #32 bit precision is full precision no quantization required
        return weights

    qmin = -(2 ** (bits - 1))   # -128
    qmax = (2 ** (bits - 1)) - 1  #127

    maximum = 0

    for row in weights:   #this will find the largest weight (by magnitude) in the entire weight matrix.
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



# 3. SOFTMAX convert the raw logits into probabilties


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

            e = math.exp((value / temperature) - largest)

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

            loss += pt * math.log((pt + 1e-9) / (ps + 1e-9))    # KL LOSS function and 1e-9 is added ot avoid log(0)

            gradient[i][j] = (ps - pt) * temperature / batch

    loss = loss / batch    # average loss 
    loss = loss * temperature * temperature   # to mkae the gradients smaller

    return loss, gradient



# 5. QUANTIZED LINEAR LAYER


class QuantizedLinearLayer:
    def __init__(self, input_size, output_size, bits):
        self.weight = random_matrix(input_size, output_size)
        self.bias = create_matrix(1, output_size)
        self.bits = bits

    def forward(self, x):

        self.input = x

        if self.bits == 32:
            w = self.weight
        else:
            w = quantize_dequantize(self.weight, self.bits)

        output = matrix_multiply(x, w)

        output = add_bias(output, self.bias)

        return output

    def backward(self, grad_output):   #the backward pass 
        input_T = transpose(self.input) 

        self.dw = matrix_multiply(input_T, grad_output) # this is the matrix mutiplication of input calcualted above and gradient output produced by the partial detivative of the model's loss or error

        self.db = create_matrix(1, len(grad_output[0]))

        for row in grad_output:

            for j in range(len(row)):
                self.db[0][j] += row[j]

    def update(self, lr):
        # update using learning rate as 
        
        for i in range(len(self.weight)):
            for j in range(len(self.weight[0])):
                self.weight[i][j] -= lr * self.dw[i][j]

        for j in range(len(self.bias[0])):
            self.bias[0][j] -= lr * self.db[0][j]




# 6. TRAINING

random.seed(0)

batch = 4
input_size = 5
output_size = 3

teacher = QuantizedLinearLayer(input_size, output_size, 32)

student = QuantizedLinearLayer(input_size, output_size, 4)

# Copy teacher weights

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