import torch
import torch.nn as nn
from torchvision import models, transforms
from lime import lime_image
from skimage.segmentation import mark_boundaries
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from inference import preprocess_image, predict

# Initialize the model
def initialize_model(num_classes, feature_extract, use_pretrained=True):
    model_ft = models.resnet18(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    return model_ft

# Function to set parameter requires grad
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def lime_interpret_image_inference(args, model):
    # Initialize LIME
    explainer = lime_image.LimeImageExplainer()

    # Path to the image you want to explain
    image_path = args.image_path
    image = preprocess_image(image_path, (224, 224)).squeeze(0).permute(1, 2, 0).numpy()
    # define a partial function to pass to lime that takes the model as the first parameter
    def predict_fn(x):
        return predict(model, x, args.device)
    # Explain the image
    explanation = explainer.explain_instance(image, predict_fn, top_labels=5, hide_color=0, num_samples=1000)
    # Get the explanation for the top class
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=10, hide_rest=False)
    img_boundry1 = mark_boundaries(temp / 255.0, mask)

    # Display the image with explanations
    plt.imshow(img_boundry1)
    plt.title('LIME Explanation')
    plt.show()