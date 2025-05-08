import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import cv2
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
import os
from pathlib import Path
import random

# Function to plot training curves
def plot_training_curves(history: dict, filename: str, title: str) -> None:
    plt.figure(figsize=(15, 5))

    # Plot training and validation accuracy values
    plt.subplot(1, 2, 1)
    plt.plot(history["accuracy"], label="Training Accuracy")
    plt.plot(history["val_accuracy"], label="Validation Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(loc="lower right")

    # Plot training and validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history["loss"], label="Training Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(loc="upper right")
    plt.suptitle(title, fontsize=16, fontweight="bold")

    plt.savefig("./plots/{}".format(filename))
    plt.tight_layout()
    plt.show()

# Function to predict categories
def predict_cats(model, dataset) -> dict:
    prediction = {}
    # Getting the true values
    true_vals = np.concatenate([label.numpy() for img, label in dataset], axis=0)
    true_vals = np.argmax(true_vals, axis=-1)
    # Add true values to the dictionary
    prediction["true"] = true_vals
    # Probability of classes
    pred_prob = model.predict(dataset)
    # Getting the predicted values
    pred_vals = np.argmax(pred_prob, axis=-1)
    # Add predicted values to the dictionary
    prediction["pred"] = pred_vals

    return prediction

# Function to show model performance evaluation
def show_evaluation_report(model, dataset: list, filename: str, matrix_title: str) -> None:
    colors = ["crest", "coolwarm", "Spectral", "flare"]
    # Dictionary of true and predicted values from the dataset
    for _set in dataset:
        prediction = predict_cats(model=model, dataset=_set)
        accuracy = accuracy_score(prediction["true"], prediction["pred"])
        plt.figure(figsize=(10, 7))
        # Confusion matrix
        ax = sns.heatmap(
            confusion_matrix(prediction["true"],prediction["pred"]),
            annot=True,
            xticklabels=_set.class_names,
            yticklabels=_set.class_names,
            cmap=random.choice(colors)
        )
        ax.set_title(matrix_title)

        plt.savefig("./plots/{}".format(filename))
        plt.show()

        print("\n\nClassification Report:\n")
        print(classification_report(prediction["true"], prediction["pred"], target_names=_set.class_names))

        print("\n\nModel Accuracy: {:.2f}".format(accuracy))

# Function for plotting the sample images
def plot_sample_images(cnames: list, cimages: dict, filename: str, title: str) -> None:
    fig, axes = plt.subplots(len(cnames), 5, figsize=(12, 8))

    for i, cat in enumerate(cnames):
        for j in range(5):
            if j < len(cimages[cat]):
                ax = axes[i, j]
                ax.imshow(cimages[cat][j] / 255.) # /255. to normalize RGB values of image
                # Turn off ticks for better display (remove axis numbers)
                ax.axis("off")
                # Placing category name at middle image
                if j == 2:
                    ax.set_title(cat)

    plt.tight_layout()
    plt.suptitle(title, fontsize=16, fontweight="bold")
    plt.savefig("./plots/{}".format(filename))
    plt.show()

# Function to get sample images from a dataset
def get_sample_images(dataset, cnames: list, n_samples: int = 5) -> dict:
    # Dictionary to hold images for each class
    class_images = {cat: [] for cat in cnames}

    for image, label in dataset.unbatch():
        cat = cnames[np.argmax(label.numpy())]
        # Take specified number of sample images per class
        if len(class_images[cat]) < n_samples:
            class_images[cat].append(image.numpy())
        if all(len(images) >= n_samples for images in class_images.values()):
            break
    return class_images

# Function to edge preprocess an image: Grayscale -> Blur -> Canny Edge Detection
def edge_enhanced_preprocessing(
    image_path: str,
    blur_kernel: tuple =(5, 5),
    canny_thresh: tuple=(100, 200),
    img_size: tuple=(256, 256)
) -> tuple:
    # Read and resize image
    image = cv2.imread(str(image_path))
    image = cv2.resize(image, img_size)

    # Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Gaussian Blur
    blurred = cv2.GaussianBlur(gray, blur_kernel, 0)

    # Canny Edge Detection
    edges = cv2.Canny(blurred, canny_thresh[0], canny_thresh[1])

    # Convert single channel to 3-channel
    edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

    return image, edges_3ch

# Function to perform edge preprocessing (experiment one) and save the processed dataset
def edge_preprocess_and_save_dataset(input_dir: str, output_dir: str) -> None:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    for split in ["train", "test", "val"]:
        input_split_dir = input_dir / split
        output_split_dir = output_dir / split

        for class_folder in input_split_dir.iterdir():
            if class_folder.is_dir():
                output_class_dir = output_split_dir / class_folder.name
                output_class_dir.mkdir(parents=True, exist_ok=True)

                for img_file in class_folder.glob("*.jpg"):
                    original, processed = edge_enhanced_preprocessing(img_file)

                    # Save the processed image
                    save_path = output_class_dir / img_file.name
                    cv2.imwrite(str(save_path), processed)

