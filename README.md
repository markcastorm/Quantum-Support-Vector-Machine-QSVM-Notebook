
# Quantum Support Vector Machine (QSVM) Notebook

## Introduction

This notebook demonstrates how to implement a Quantum Support Vector Machine (QSVM) using Qiskit, a quantum computing framework. QSVM is a quantum variant of the classical Support Vector Machine (SVM) algorithm. By leveraging quantum computing resources, QSVM can find decision boundaries in higher-dimensional spaces, potentially improving the performance of SVM for certain datasets.


## Installation

Ensure you have the necessary libraries installed:

```python
  pip install qiskit pylatexenc qiskit-aer qiskit-algorithms qiskit-machine-learning

```
    
## Quantum Feature Maps

Quantum feature maps are used to map classical input data points to higher-dimensional quantum states. The ZZFeatureMap is a specific type of feature map that uses ZZ-interactions between qubits to achieve this mapping.

```python
import numpy as np
import matplotlib.pyplot as plt
from qiskit.circuit.library import ZZFeatureMap

num_qubits = 4
x = np.random.random(num_qubits)
data = ZZFeatureMap(feature_dimension=num_qubits, reps=1, entanglement="linear")
data.assign_parameters(x, inplace=True)
data.decompose().draw("mpl", style="iqx", scale=1.4)

```

In this example, a random input vector x is created and assigned to the parameters of the ZZFeatureMap circuit. The circuit is then decomposed and drawn.

## Building a Quantum Kernel

A quantum kernel is evaluated using quantum circuits. The kernel trick allows us to compute the dot product of two vectors in a higher-dimensional space.

```python
  from qiskit import BasicAer, transpile, QuantumCircuit

backend = BasicAer.get_backend("qasm_simulator")
shots = 1024
dimension = 2
feature_map = ZZFeatureMap(dimension, reps=1)

def evaluate_kernel(x_i, x_j):
    circuit = QuantumCircuit(dimension)
    circuit.compose(feature_map.assign_parameters(x_i), inplace=True)
    circuit.compose(feature_map.assign_parameters(x_j).inverse(), inplace=True)
    circuit.measure_all()
    
    transpiled = transpile(circuit, backend)
    counts = backend.run(transpiled, shots=shots).result().get_counts()
    return counts.get("0" * dimension, 0) / shots

# Example usage with sample data
from data_generators import circle

X, y = circle()
evaluate_kernel(X[2], X[3])

```

Here, the evaluate_kernel function constructs a quantum circuit, applies the feature map to two input vectors, and measures the resulting state. The dot product in the higher-dimensional space is estimated by the measurement outcomes.

## Using Qiskit Machine Learning

Qiskit Machine Learning provides higher-level abstractions for implementing QSVM.


```python
import qiskit
from qiskit.providers.aer import AerSimulator
from qiskit.circuit.library import ZZFeatureMap
from qiskit.primitives import Sampler
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC

backend = BasicAer.get_backend("qasm_simulator")
feature_map = ZZFeatureMap(dimension, reps=1)
sampler = Sampler()
fidelity = ComputeUncompute(sampler=sampler)
kernel = FidelityQuantumKernel(feature_map=feature_map, fidelity=fidelity)

# Evaluate the kernel for two data points
kernel_value = kernel.evaluate(X[2], X[3])


```

This code sets up a fidelity-based quantum kernel and evaluates it for two data points from the circle dataset.


## Classification

We use the circle dataset for classification. Points are labeled as 1 if they are outside a radius of 0.6 and -1 if they are inside.

```python
points, labels = circle()
colors = ["crimson" if label == 1 else "royalblue" for label in labels]

plt.figure(figsize=(6, 6))
plt.scatter(points[:, 0], points[:, 1], c=colors)
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.show()


```
The dataset is visualized with points colored based on their labels.

## Training QSVM

We train a quantum SVM using the quantum kernel.

```python
from sklearn.svm import SVC

qsvm = SVC(kernel=kernel.evaluate)
qsvm.fit(points, labels)
predicted = qsvm.predict(points)


```


The quantum kernel is used as the kernel function for the SVM, and the model is trained on the dataset.

## Visualization of Results

Finally, we visualize the classification results.

```python
markers = [
    "o" if label == predicted_label else "x"
    for label, predicted_label in zip(labels, predicted)
]

plt.figure(figsize=(6, 6))
for point, marker, color in zip(points, markers, colors):
    plt.scatter(point[0], point[1], c=color, marker=marker)
plt.show()


```

Correctly classified points are marked with 'o', while misclassified points are marked with 'x'.


## Conclusion

This notebook demonstrates the implementation of a QSVM using Qiskit. By utilizing quantum feature maps and quantum kernels, we can potentially improve the performance of SVM on certain datasets. The notebook covers the steps from feature mapping to kernel evaluation and classification.
References

    Qiskit Documentation
    Qiskit Machine Learning
    Scikit-learn Documentation

