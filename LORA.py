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

def transpose(M):
    return M.T


zero_matrix = create_zero_matrix(2,3)
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
        self.last_x = None
        self.last_lora_A = None
        
        # Gradients
        self.grad_A = None
        self.grad_B = None
        
        
        
        
        # x can be a single vector of shape (in_features, 1) 
        # or a batch of data of shape (batch_size, in_features).
    def forward(self, x):

        self.last_x = x  # Cache input for backward pass
        
        # Standard frozen pathway: h_base = x * W0^T
        base_output = np.dot(x, self.W0.T)
        
        # LoRA pathway: delta_h = ((x * A^T) * B^T) * scaling
        self.last_lora_A = np.dot(x, self.A.T)  # Cache intermediate state
        lora_output = np.dot(self.last_lora_A, self.B.T) * self.scaling
        
        # Combined output
        return base_output + lora_output


    def backward(self, grad_output):
    # Computes the analytical gradients for A and B using the chain rule.
    # grad_output shape: (batch_size, out_features)
    # Factor the scaling constant directly into the incoming gradient
        scaled_grad = grad_output * self.scaling
        
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
    #Applies basic Stochastic Gradient Descent (SGD) to update A and B
        self.A -= lr * self.grad_A
        self.B -= lr * self.grad_B

    def get_merged_weights(self):
        # Computes and returns the consolidated matrix: W0 + scaling * (B @ A)
        return self.W0 + (self.scaling * np.dot(self.B, self.A))
