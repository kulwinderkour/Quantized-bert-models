class is basically userdefined template or blueprint to create objects

activation function is calcuated before makeing predictions to introduce the non linearity so that the matrix learns the complex data 

calibration is the prediction and check how much model aligns with the real results
is the process of measuring the values produced by a trained ai model before converting them to INT8
 

activation calibration is the process of passing a small representative dataset through a trained model to calculate the activbations 


scale is how much one integer represents in floating point
zero point shifts the mapping 

activation is the output of the neuron used as the input for the next layer





# DATASET COMPARISONS

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