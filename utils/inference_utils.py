import torch
import matplotlib.pyplot as plt
from torchvision import transforms


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
def preprocess_image(image, img_shape):
    preprocess = transforms.Compose(
        [
            transforms.Resize(img_shape),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    image = preprocess(image).unsqueeze(0)
    return image


# Function to predict
def predict(model, image):
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
    return preds


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
