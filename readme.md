# activation function
is calcuated before makeing predictions to introduce the non linearity so that the matrix learns the complex data 

# calibration
is the prediction and check how much model aligns with the real results
is the process of measuring the values produced by a trained ai model before converting them to INT8
 

# activation calibration
is the process of passing a small representative dataset through a trained model to calculate the activbations 

# activation 
is the output of the neuron used as the input for the next layer is present

scale is how much one integer represents in floating point
zero point shifts the mapping 
class is basically userdefined template or blueprint to create objects






# DATASET COMPARISONS

dataset comaprisons of PTQ,QAT, LoRA, QLoRA and QAD trained on IMBD movies dataset


    1. PTQ
    --- FP32 Evaluation Metrics ---
    Accuracy : 0.6621  
    Precision: 0.4696
    Recall   : 0.1602
    Inference Time: 0.0535 seconds

    --- Quantized (INT8) Evaluation Metrics ---
    Accuracy : 0.6606
    Precision: 0.4638
    Recall   : 0.1617
    Inference Time: 0.3526 seconds


        
    2. QTA 

    
    Total Test Samples:    1000
    --- Classification Performance ---
    Accuracy:              99.00%
    Precision:             99.61%
    Recall:                98.44%
    F1-Score:              99.02%
    Inference Time: 0.0011 seconds
    Confusion Matrix (TN  FP / FN  TP):
     [485  2]
     [8  505]
    
    --- Inference Time Performance ---
    Total Execution Time:  1.079 ms
    Avg Latency / Sample:  1.079 µs
    Throughput:            927,042 Inferences/sec
    
    
    3. LORA
    
    
    --- FP32 Evaluation Metrics ---
    Accuracy : 0.5113
    Precision: 0.5119
    Recall   : 0.5166
    Inference Time: 0.0014 seconds
    
    --- Quantized (INT8) Evaluation Metrics ---
    Accuracy : 0.5107
    Precision: 0.5112
    Recall   : 0.5166
    Inference Time: 0.0012 seconds
    
    
    4. QLORA 
    
    --- Starting QLoRA Fine-Tuning ---
    Base Weight Matrix: 3x8 (Stored in 4-bit NF4)
    LoRA Adapters: A(2x8), B(3x2) in FP32
    
    Epoch  1/10 | Training Loss: 0.038026
    Epoch  2/10 | Training Loss: 0.038026
    Epoch  3/10 | Training Loss: 0.038025
    Epoch  4/10 | Training Loss: 0.038024
    Epoch  5/10 | Training Loss: 0.038024
    Epoch  6/10 | Training Loss: 0.038023
    Epoch  7/10 | Training Loss: 0.038023
    Epoch  8/10 | Training Loss: 0.038022
    Epoch  9/10 | Training Loss: 0.038022
    Epoch 10/10 | Training Loss: 0.038021
    
    --- Evaluating Fine-Tuned Model ---
    
    QLoRA EVALUATION METRICS             
    
    Total Test Predictions: 1500
    
    --- Classification Performance ---
    Accuracy : 0.5127 (51.27%)
    Precision: 0.5133 (51.33%)
    Recall   : 0.5140 (51.40%)
    F1-Score : 0.5136 (51.36%)
    
    Confusion Matrix (TN FP / FN TP):
     [383  366]
     [365  386]
    
    --- Inference Time Performance ---
    Total Inference Time: 0.0012 seconds
    Avg Latency / Sample: 2.382 µs
    
    
    
    5. QAD
    
    Training QAD...
    
    Epoch 1 Loss = 0.000721
    Epoch 2 Loss = 0.000719
    Epoch 3 Loss = 0.000718
    Epoch 4 Loss = 0.000717
    Epoch 5 Loss = 0.000716
    Epoch 6 Loss = 0.000716
    Epoch 7 Loss = 0.000716
    Epoch 8 Loss = 0.000233
    Epoch 9 Loss = 0.00023
    Epoch 10 Loss = 0.000228
    
    --- Evaluating Fine-Tuned Student Model ---
        QAD EVALUATION METRICS
    Total Predictions Assessed: 12
    
    --- Classification Performance ---
    Accuracy : 1.0000 (100.00%)
    Precision: 1.0000 (100.00%)
    Recall   : 1.0000 (100.00%)
    F1-Score : 1.0000 (100.00%)
    
    Confusion Matrix (TN FP / FN TP):
     [5  0]
     [0  7]
    
    --- Inference Time Performance ---
    Total Inference Time: 0.000161 seconds
    Avg Latency / Sample: 40.175 µs

  

