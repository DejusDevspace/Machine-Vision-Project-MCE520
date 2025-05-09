# Thought Process

## Points to document

    * The vanilla cnn training was unstable, so we had to reduce the learning rate of the optimizer (Adam) from default (0.001) to 0.0001 (1e-4).
    * The training in experiment one with the original dataset converged faster and it is noticeable that the edge detection with opencv was not very accurate.

## Documentation Notes

    * Documentation would be by experiments which would cover different preprocessing techniques.
    * Write (if available) the meaning, effect, and results of each preprocessing technique applied to data - with relevant code snippets.

# Experiments

## Experiment 1

Transfer learning using ResNet50 Pretrained CNN without any data augmentation or preprocessing steps (original data).

## Experiment 2

Title: Edge Detection...
Grayscale + Gaussian Blur + Canny Edge Detection.

## Experiment 3

Title: Shape Feature Classification via Contour Analysis
Grayscale → Histogram Equalization → Contour Detection: Use an SVM for classification after preprocessing
