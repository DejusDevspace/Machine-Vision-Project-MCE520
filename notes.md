# Thought Process

## Points to document

- The vanilla cnn training was unstable, so we had to reduce the learning rate of the optimizer (Adam) from default (0.001) to 0.0001 (1e-4).
- The training in experiment one with the original dataset converged faster and it is noticeable that the edge detection with opencv was not very accurate.

## Documentation Notes

- Documentation would be by experiments which would cover different preprocessing techniques.
- Write (if available) the meaning, effect, and results of each preprocessing technique applied to data - with relevant code snippets.
- Note what each effect captures...e.g, "the use of x technique helps the model to generalize more to the data and it mimics real-world noise or data variations".
- Total number of images across the project. Data structure across all the sets (training, test, and val), and number of images per class.

# Experiments

## Experiment 1

**Title**: Base Case

Transfer learning using ResNet50 Pretrained CNN without any data augmentation or preprocessing steps (original data).

## Experiment 2

**Title**: Edge Detection...

Grayscale + Gaussian Blur + Canny Edge Detection.

## Experiment 3

**Title**: Shape Feature Classification via Contour Analysis

Grayscale → Histogram Equalization → Contour Detection: Use an SVM for classification after preprocessing

## Experiment 4

**Title**: Data Augmentation Pipeline

Using data augmentation techniques like rotation, flipping, zooming...
