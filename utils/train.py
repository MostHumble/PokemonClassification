from torchvision import models
import torch.nn as nn
from tqdm import tqdm
import torch
import mlflow
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Define the training loop
def train_one_epoch(model, trainloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(trainloader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track loss and accuracy
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / len(trainloader)
    epoch_accuracy = 100.0 * correct / total
    return epoch_loss, epoch_accuracy


# Define the evaluation loop
@torch.no_grad()
def evaluate(model, testloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    for images, labels in tqdm(testloader, desc="Evaluating", leave=False):
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Track loss and accuracy
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        # Collect all labels and predictions for metrics
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

    epoch_loss = running_loss / len(testloader)

    # Calculate accuracy, precision, recall, and F1-score
    epoch_accuracy = accuracy_score(all_labels, all_predictions, normalize=True) * 100
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')

    return epoch_loss, epoch_accuracy, precision, recall, f1


# Define the pipeline
def train_and_evaluate(
    model, trainloader, testloader, criterion, optimizer, device, epochs, use_mlflow=False
):
    """
    Train and evaluate the model.

    Args:
        model (nn.Module): The neural network model.
        trainloader (DataLoader): DataLoader for training data.
        testloader (DataLoader): DataLoader for test data.
        criterion (nn.Module): Loss function.
        optimizer (optim.Optimizer): Optimizer.
        device (torch.device): Device to train on ('cuda' or 'cpu').
        epochs (int): Number of epochs to train.

    Returns:
        dict: Training and evaluation statistics.
    """
    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": [], "precision": [], "recall": [], "f1": []}

    model.to(device)

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        # Train for one epoch
        train_loss, train_acc = train_one_epoch(
            model, trainloader, criterion, optimizer, device
        )
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")

        # Evaluate the model
        test_loss, test_acc, precision, recall, f1 = evaluate(model, testloader, criterion, device)
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")

        # Save statistics
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["precision"].append(precision)
        history["recall"].append(recall)
        history["f1"].append(f1)
    
        if use_mlflow:
            mlflow.log_metric("epoch", epoch)
            mlflow.log_metric("train_loss", train_loss)
            mlflow.log_metric("train_acc", train_acc)
            mlflow.log_metric("test_loss", test_loss)
            mlflow.log_metric("test_acc", test_acc)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1", f1)
    return history


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(
    model_name, num_classes, feature_extract=True, use_pretrained=True
):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(
            512, num_classes, kernel_size=(1, 1), stride=(1, 1)
        )
        model_ft.num_classes = num_classes

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft
