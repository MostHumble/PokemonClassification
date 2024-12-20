import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import os
import random
from utils.data import CLASS_NAMES

# Function to find correctly and incorrectly classified images
def find_images(dataloader, model, device, num_correct, num_incorrect):
    correct_images = []
    incorrect_images = []
    correct_labels = []
    incorrect_labels = []
    correct_preds = []
    incorrect_preds = []

    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            for i in range(images.size(0)):
                if preds[i] == labels[i] and len(correct_images) < num_correct:
                    correct_images.append(images[i].cpu())
                    correct_labels.append(labels[i].cpu())
                    correct_preds.append(preds[i].cpu())
                elif preds[i] != labels[i] and len(incorrect_images) < num_incorrect:
                    incorrect_images.append(images[i].cpu())
                    incorrect_labels.append(labels[i].cpu())
                    incorrect_preds.append(preds[i].cpu())

                if (
                    len(correct_images) >= num_correct
                    and len(incorrect_images) >= num_incorrect
                ):
                    break
            if (
                len(correct_images) >= num_correct
                and len(incorrect_images) >= num_incorrect
            ):
                break

    return (
        correct_images,
        correct_labels,
        correct_preds,
        incorrect_images,
        incorrect_labels,
        incorrect_preds,
    )

def find_images_from_path(data_path, model, device, num_correct=2, num_incorrect=2, label=None):
    correct_images_paths = []
    incorrect_images_paths = []
    correct_labels = []
    incorrect_labels = []

    label_to_idx = {label: idx for idx, label in enumerate(CLASS_NAMES)}

    model.eval()
    # First collect available images for the specified label or all labels
    label_images = {}
    if label:
        if os.path.isdir(os.path.join(data_path, label)):
            label_path = os.path.join(data_path, label)
            label_images[label] = [os.path.join(label_path, img) for img in os.listdir(label_path)]
    else:
        for label in os.listdir(data_path):
            label_path = os.path.join(data_path, label)
            if not os.path.isdir(label_path):
                continue
            label_images[label] = [os.path.join(label_path, img) for img in os.listdir(label_path)]

    # Randomly process images until we have enough samples
    with torch.no_grad():
        while len(correct_images_paths) < num_correct or len(incorrect_images_paths) < num_incorrect:
            # Randomly select a label that still has unprocessed images
            available_labels = [l for l in label_images if label_images[l]]
            if not available_labels:
                break
                
            selected_label = random.choice(available_labels)
            image_path = random.choice(label_images[selected_label])
            label_images[selected_label].remove(image_path)  # Remove the selected image
            
            image = preprocess_image(image_path, (224, 224)).to(device)
            label_idx = label_to_idx[selected_label]
            
            outputs = model(image)
            _, pred = torch.max(outputs, 1)

            if pred == label_idx and len(correct_images_paths) < num_correct:
                correct_images_paths.append(image_path)
                correct_labels.append(label_idx)
            elif pred != label_idx and len(incorrect_images_paths) < num_incorrect:
                incorrect_images_paths.append(image_path)
                incorrect_labels.append(label_idx)

    save_images_by_class(correct_images_paths, correct_labels, incorrect_images_paths, incorrect_labels)

def save_images_by_class(correct_images_paths, correct_labels, incorrect_images_paths, incorrect_labels):
    # Create root directories for correct and incorrect classifications
    for class_name in CLASS_NAMES:
        os.makedirs(os.path.join('predictions', class_name, 'correct'), exist_ok=True)
        os.makedirs(os.path.join('predictions', class_name, 'mistake'), exist_ok=True)

    # Save correctly classified images
    for img_path, label in zip(correct_images_paths, correct_labels):
        class_name = CLASS_NAMES[label]
        img_name = os.path.basename(img_path)
        destination = os.path.join('predictions', class_name, 'correct', img_name)
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        Image.open(img_path).save(destination)

    # Save incorrectly classified images
    for img_path, label in zip(incorrect_images_paths, incorrect_labels):
        class_name = CLASS_NAMES[label]
        img_name = os.path.basename(img_path)
        destination = os.path.join('predictions', class_name, 'mistake', img_name)
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        Image.open(img_path).save(destination)

def show_samples(dataloader, model, device, num_correct=3, num_incorrect=3):
    # Get some correctly and incorrectly classified images
    (
        correct_images,
        correct_labels,
        correct_preds,
        incorrect_images,
        incorrect_labels,
        incorrect_preds,
    ) = find_images(dataloader, model, device, num_correct, num_incorrect)
    # Display the results in a grid
    fig, axes = plt.subplots(
        num_correct + num_incorrect, 1, figsize=(10, (num_correct + num_incorrect) * 5)
    )

    for i in range(num_correct):
        axes[i].imshow(correct_images[i].permute(1, 2, 0))
        axes[i].set_title(
            f"Correctly Classified: True Label = {correct_labels[i]}, Predicted = {correct_preds[i]}"
        )
        axes[i].axis("off")

    for i in range(num_incorrect):
        axes[num_correct + i].imshow(incorrect_images[i].permute(1, 2, 0))
        axes[num_correct + i].set_title(
            f"Incorrectly Classified: True Label = {incorrect_labels[i]}, Predicted = {incorrect_preds[i]}"
        )
        axes[num_correct + i].axis("off")

    plt.tight_layout()
    plt.show()


# Function to preprocess image
def preprocess_image(image_path, img_shape):

    # Load the image using PIL
    image = Image.open(image_path)

    # Apply preprocessing transformations
    preprocess = transforms.Compose([
        transforms.Resize(img_shape),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = preprocess(image).unsqueeze(0)

    return image


# Function to predict
def predict(model, image):
    model.eval()
    with torch.no_grad():
        outputs = model(image)
    return outputs


# Function to get model predictions for LIME
def batch_predict(model, images, device):
    model.eval()
    batch = torch.stack(
        tuple(preprocess_image(image, (224, 224)) for image in images), dim=0
    )
    batch = batch.to(device)
    logits = model(batch)
    probs = torch.nn.functional.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()
