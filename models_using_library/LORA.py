import math
import random
import numpy as np

# for initialztion of matrix with 0
def create_zero_matrix(rows,cols):
    return np.zeros((rows,cols))

# create gaussian matrix for the random values
def gaussain_matrix(rows,cols,std=0.1):
    return np.random.normal(loc=0.0,scale=std,size=(rows,cols))  # loc specific mean(the center of the distribution)  and scale specifies the standard deviation and size is the dimensions


def matrix_multiplication(A,B):
    return np.dot(A,B)
# use the dot product for the two matrix multiplication

def transpose(M):
    return M.T


zero_matrix = create_zero_matrix(2,3)  # initalize the zero matrix with 2 rows nad 3 columns 
print("Zero matrix:\n")
print(zero_matrix)
print("\n")

# create two gaussian distribution

A = gaussain_matrix(2,3)
B = gaussain_matrix(3,2)

print("Matrix A\n:")
print(A)
print("\n")

print("Matrix B:\n")
print(B)
print("\n")
result = matrix_multiplication(A,B)


print("A x B:\n")
print(result)



    
    
   
   
# LORALINEAR LAYER

np.random.seed(42)   # This fixes the sequence of random numbers.

class LORALinear:
    def __init__(self, in_features, out_features, rank=4, alpha=8.0):
        self.in_features = in_features  # input features.
        self.out_features = out_features # output features 
        self.rank = rank  #Rank decides how much information the low-rank update can represent.
        self.alpha = alpha  # hyperparameter Alpha controls how strongly the LoRA update affects the frozen model.
        self.scaling = alpha / rank  # The scaling factor keeps the magnitude of the update stable across different rank values.
        
        # 1. Base Pretrained Weight Matrix (Frozen)  Frozen Weight Matrix
        # Shape: (out_features, in_features)
        self.W0 = np.random.randn(out_features, in_features) * 0.5
        
        # 2. Trainable LoRA Matrices
        # Matrix A shape: (rank, in_features) -> Initialized with random Gaussian
        # why random - To break symmetry and provide learnable directions in the low-rank space.
        self.A = np.random.randn(rank, in_features) * 0.1
        # Matrix B shape: (out_features, rank) -> Initialized to zero
        self.B = np.zeros((out_features, rank))
        
        
        
        # Placeholders to cache inputs during the forward pass for backpropagation
        # During forward pass the model computes xAT During backward pass, we need that same value to calculate gradients. Instead of recomputing, we store it.basically for the caching

        self.last_x = None   # we stores teh last input x that comes during forward pass
        self.last_lora_A = None  # stores the matrix A's intermediate output 
        
        # Gradients
        self.grad_A = None   # lora A gradients
        self.grad_B = None   # lora B graidents
        
        
        
    
        # x can be a single vector of shape (in_features, 1) 
        # or a batch of data of shape (batch_size, in_features).
    def forward(self, x):

        self.last_x = x  # it store the input of forward pass Cache for backward pass
        
        # Standard frozen pathway: h_base = x * W0^T
        base_output = np.dot(x, self.W0.T)
        #
        
        # LoRA pathway: delta_h = ((x * A^T) * B^T) * scaling
        self.last_lora_A = np.dot(x, self.A.T)  # Cache intermediate state
        lora_output = np.dot(self.last_lora_A, self.B.T) * self.scaling   # scailing is class variable that stores the scialing factor
        # this stores the lora matrix A and B
        # Combined output
        return base_output + lora_output


    def backward(self, grad_output): # calcuate the gradient of the lora of A and B uisng chain rule
        
    # grad_output shape: (batch_size, out_features)
    # Factor the scaling constant directly into the incoming gradient
        scaled_grad = grad_output * self.scaling  # as we have multiply the scailing wiht the output of forward pass and same happens in backward also 
        
        # 1. Gradient for B: dL/dB = (scaled_grad)^T * (x * A^T)
        # Dimensions: (out_features, batch_size) @ (batch_size, rank) -> (out_features, rank)
        self.grad_B = np.dot(scaled_grad.T, self.last_lora_A)
        
        # 2. Gradient for A: dL/dA = [(scaled_grad * B) * x]
        # Dimensions: (batch_size, out_features) @ (out_features, rank) -> (batch_size, rank)
        grad_intermediate = np.dot(scaled_grad, self.B)
        
        # Dimensions: (rank, batch_size) @ (batch_size, in_features) -> (rank, in_features)
        self.grad_A = np.dot(grad_intermediate.T, self.last_x)
        
        return None


    def update_weights(self, lr):
        self.A -= lr * self.grad_A   # then we use the learnign rate to update the weights 
        self.B -= lr * self.grad_B

    def get_merged_weights(self):
        # Computes and returns the consolidated matrix: W0 + scaling * (B @ A)
        return self.W0 + (self.scaling * np.dot(self.B, self.A))




# ==========================================
# Testing the LoRA Layer
# ==========================================

# Create a LoRA layer
layer = LORALinear(
    in_features=3,
    out_features=2,
    rank=2,
    alpha=4
)

print("\n========== LoRA Layer ==========\n")

print("Frozen Weight Matrix W0:")
print(layer.W0)

print("\nLoRA Matrix A:")
print(layer.A)

print("\nLoRA Matrix B:")
print(layer.B)

# Create an input batch
# batch_size = 2
# in_features = 3

x = np.array([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]
])

print("\nInput X:")
print(x)

# Forward Pass
output = layer.forward(x)

print("\nForward Output:")
print(output)

# Dummy Gradient coming from next layer
grad_output = np.ones((2, 2))

print("\nGradient from Next Layer:")
print(grad_output)

# Backward Pass
layer.backward(grad_output)

print("\nGradient of A:")
print(layer.grad_A)

print("\nGradient of B:")
print(layer.grad_B)

# Update Weights
learning_rate = 0.01

layer.update_weights(learning_rate)

print("\nUpdated Matrix A:")
print(layer.A)

print("\nUpdated Matrix B:")
print(layer.B)

# Merge LoRA into W0
merged = layer.get_merged_weights()

print("\nMerged Weight Matrix:")
print(merged)





# ************************
# let's test the layer of data  size = 4, input features =8 , output features = 3
X_train = np.random.randn(4,8)
Y_target = np.random.randn(4,3)

lora_layer = LORALinear(in_features=8,out_features=3, rank=2,alpha=4.0)
learning_rate=0.01  # how much parameter change after each update

print("--Training Process--")
for epoch in range(1,6):   # epoch means one complete pass through training data
    prediction = lora_layer.forward(X_train)   # this sends the input through the lora layer 
    error = prediction - Y_target   # calcutate hte erorr prediction -error

    loss = np.mean(0.5* (error*2))
    print(f"Epoch {epoch} | loss {Loss:6f}") 
    grad_output = error/X_train.shape[0]
    
    
    lora_layer.backward(grad_output)

    lora_layer.update_weights(learning_rate)

print("weight for deployment")

W_final = lora_layer.get_merged_weights()
print(f"Original base weights shape:{lora_layer.W0.shape}")
print(f"Merged downstream weights shape: {W_final.shape}")


