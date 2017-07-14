MNIST CNN
---
Convolutional Neural Network built on Tensorflow for `MNIST Dataset`

Overview
---
This model is built for informative purpose rather than a state-of-art neural netfork.

###Layers
```
Input -> Conv -> ReLU -> Pool -> Conv -> ReLU -> Pool -> FullC -> FullC -> Output
```

* **Input**: An `[28 x 28 x 1]` image of a handwriten digit (grayscale).
* **Conv1**: First Convolutional Layer, 32 filters with `[5 x 5 x 1]` shape.
* **ReLU1**: Replace every negative value in `Conv1` with `0`.
* **Pool1**: Max Pooling Layer `[2 x 2]` with stride = `2`/
* **Conv2**: Second Convolutional Layer, 64 filters with `[5 x 5 x 32]` shape.
* **ReLU2**: Replace every negative value in `Conv2` with `0`.
* **Pool2**: Max Pooling Layer `[2 x 2]` with stride = `2`.
* **FullyConnected1**: Fully Connected Layer. 1024 neurons with `[7 * 7 * 4]` shape.
* **FullyConnected2**: Fully Connected Layer. 10 ouput neurons for each class.



LICENSE
---
[MIT](LICENSE)