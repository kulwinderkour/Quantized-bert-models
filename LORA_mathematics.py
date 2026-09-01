import random    # for generating the random numbers    


# MATRIX OPERATIONS 

def create_zero_matrix(rows, cols):   # we will initalize the zero matrix
    matrix = []
    for i in range(rows):
        row = []
        for j in range(cols):
            row.append(0.0)
        matrix.append(row)
    return matrix


def gaussian_matrix(rows, cols, std=0.1):   # initalize the one matrix with the random values using gaussian distrubution
    matrix = []
    for i in range(rows):
        row = []
        for j in range(cols):
            row.append(random.gauss(0, std))
        matrix.append(row)
    return matrix


def transpose(matrix):    # transpose is used in gradients and matrix mutiplication (convert the rows into cols viceversa)
    rows = len(matrix)   
    cols = len(matrix[0])

    result = create_zero_matrix(cols, rows)

    for i in range(rows):
        for j in range(cols):
            result[j][i] = matrix[i][j]

    return result


def matrix_multiply(A, B):
    #a = 2*3 , b = 3*4    (col of a must be equal to the row of b)

    rows_A = len(A)  # 2
    cols_A = len(A[0]) # 3

    rows_B = len(B)  # 3 
    cols_B = len(B[0]) # 4

    result = create_zero_matrix(rows_A, cols_B)

    for i in range(rows_A):
        for j in range(cols_B):

            total = 0

            for k in range(cols_A):
                total += A[i][k] * B[k][j]

            result[i][j] = total

    return result


def matrix_add(A, B):   # simply add two matrics A and B

    rows = len(A)
    cols = len(A[0])

    result = create_zero_matrix(rows, cols)

    for i in range(rows):
        for j in range(cols):
            result[i][j] = A[i][j] + B[i][j]

    return result


def scalar_multiply(matrix, scalar):  # multiply with the scaling factor 

    rows = len(matrix)
    cols = len(matrix[0])

    result = create_zero_matrix(rows, cols)

    for i in range(rows):
        for j in range(cols):
            result[i][j] = matrix[i][j] * scalar

    return result


def matrix_subtract(A, B):   #Error=Prediction−Target

    rows = len(A)
    cols = len(A[0])

    result = create_zero_matrix(rows, cols)

    for i in range(rows):
        for j in range(cols):
            result[i][j] = A[i][j] - B[i][j]

    return result


def print_matrix(name, matrix):
    print(name)
    for row in matrix:
        print(row)
    print()


# LoRA Layer

class LORALinear:

    def __init__(self, in_features, out_features, rank=2, alpha=4):  # this is constrcutor

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Frozen pretrained weights
        self.W0 = gaussian_matrix(out_features, in_features, 0.5)

        # LoRA matrices
        self.A = gaussian_matrix(rank, in_features, 0.1)

        self.B = create_zero_matrix(out_features, rank)

        self.last_x = None   # store hte input of the last element
        self.last_lora = None  # store the result of the lora produced during the forward pass and often required later for the backpropogatin
        self.grad_A = None   
        self.grad_B = None


    # Forward


    def forward(self, x):

        self.last_x = x   # store hte input

        base = matrix_multiply(x, transpose(self.W0))     # X * W0

        self.last_lora = matrix_multiply(x, transpose(self.A))  #X * A  (input passes through the pretrained weights to produce the base output)

        lora = matrix_multiply(self.last_lora, transpose(self.B)) # X * B
        # A and B are mulitplied with input (x) to produce the small adaptations
        lora = scalar_multiply(lora, self.scaling)
        # Y = X*W0 + α(X * A * B )​ => Y = X(W0 ​+ αBA)​

        return matrix_add(base, lora)


    # Backward pass  this the learning phase when the model produce an output it will compare the prediction with the correct answer

    def backward(self, grad_output):

        scaled = scalar_multiply(grad_output, self.scaling)    # pred = 8, result =10 so the erorr is (10-8)^2 = 4 then the derivative of loss wrt ot preidtion (x-y)^2 = 2(pred-result) = 2(8-10) = -2 is the grad output  

        self.grad_B = matrix_multiply(transpose(scaled),  # this is the multiplication of the scaled
                                      self.last_lora)    #output and the last_lora(output of the forward pass)

        grad_intermediate = matrix_multiply(scaled,  # it is the error signal after it passed through 
                                            self.B) #matrix B and act as a bridge between A and B

        self.grad_A = matrix_multiply(transpose(grad_intermediate),
                                      self.last_x)   

    # Update

    def update(self, lr):  # this function update the trainable lora matrics  A and B
         # New Weight= Old Weight− LearningRate × Gradient
        for i in range(len(self.A)):
            for j in range(len(self.A[0])):
                self.A[i][j] -= lr * self.grad_A[i][j]

        for i in range(len(self.B)):
            for j in range(len(self.B[0])):
                self.B[i][j] -= lr * self.grad_B[i][j]

   
    # Merge

    def merged_weights(self):

        BA = matrix_multiply(self.B, self.A)

        BA = scalar_multiply(BA, self.scaling)

        return matrix_add(self.W0, BA)


# Generate Training Data

random.seed(42) 

X = gaussian_matrix(4, 8)   # random matrix

Y = gaussian_matrix(4, 3)

layer = LORALinear( # function called and stored in a layer variable and pass these parameters
    in_features=8,
    out_features=3,
    rank=2,
    alpha=4
)

learning_rate = 0.01


# Training

for epoch in range(5):

    prediction = layer.forward(X)   #Pass the input X through the LoRA layer and store the output (prediction) in the variable prediction

    error = matrix_subtract(prediction, Y)   # error= prediction - result

    # Mean Squared Error

    total = 0

    rows = len(error)
    cols = len(error[0])

    for i in range(rows):
        for j in range(cols):
            total += error[i][j] * error[i][j]   # square each error and add them error = [-2,1] = (-2)^2 + 1 = 5 (total)

    loss = total / (rows * cols)

    print("Epoch", epoch + 1, "Loss =", round(loss, 6))

    grad = create_zero_matrix(rows, cols)

    for i in range(rows):
        for j in range(cols):
            grad[i][j] = error[i][j] / rows

    layer.backward(grad)

    layer.update(learning_rate)


# Final Weights

merged = layer.merged_weights()

print()
print_matrix("Merged Weights", merged)