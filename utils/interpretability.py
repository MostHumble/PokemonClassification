from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from utils.inference_utils import predict
import os
import torch
from PIL import Image
import numpy as np


def unnormalize(image):
    # Make sure the image is on the correct dtype and device
    # Convert mean and std to torch tensors with the correct dtype
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)  # Use torch.float32
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)    # Use torch.float32

    # If the image is a PyTorch tensor, ensure it has the same dtype
    if isinstance(image, torch.Tensor):
        image = image * std + mean
    else:
        image = torch.tensor(image, dtype=torch.float32) * std + mean  # Convert to torch if necessary

    return image



def lime_interpret_image_inference(args, model, image, device):
    # prepare the image
    def prepare_for_plot(image): return unnormalize(image).cpu().numpy()
    # Remove batch dimension and Rearrange dimensions to (H, W, C)
    image = image.squeeze(0).permute(1, 2, 0)  # From From [1, 3, 224, 224] to [224, 224, 3]
    
    # Convert to NumPy array
    image_np = image.cpu().numpy()  # Ensure the tensor is on the CPU

    # Initialize LIME explainer
    explainer = lime_image.LimeImageExplainer()

    # Define the prediction function
    def predict_fn(x):
        # Convert (B, H, W, C) to PyTorch tensor (B, C, H, W)
        x_tensor = torch.tensor(x).permute(0, 3, 1, 2).to(device)
        preds = model(x_tensor)
        return preds.detach().cpu().numpy()

    # Run LIME explanation
    explanation = explainer.explain_instance(
        image_np,
        predict_fn,
        top_labels=5,
        hide_color=0,
        num_samples=5000
    )

    # Get the mask for the top predicted class
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        num_features=10,
        hide_rest=False
    )

    # Create a 2x2 subplot
    fig, axs = plt.subplots(2, 2, figsize=(15, 15))

    # Plot the original image
    axs[0, 0].imshow(prepare_for_plot(image))
    axs[0, 0].set_title("Original Image")

    # Plot the feature that contributed the most positively
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=True)
    axs[0, 1].imshow(prepare_for_plot(mark_boundaries(temp, mask)))
    axs[0, 1].set_title("Top Positive Features")

    # Plot the features that contributed the most positively and negatively
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=1000, hide_rest=False, min_weight=0.1)
    axs[1, 0].imshow(mark_boundaries(prepare_for_plot(temp), mask))
    axs[1, 0].set_title("Top Positive and Negative Features")

    # Plot a heatmap of the features
    ind = explanation.top_labels[0]
    dict_heatmap = dict(explanation.local_exp[ind])
    heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)
    im = axs[1, 1].imshow(heatmap, cmap='RdBu', vmin=-heatmap.max(), vmax=heatmap.max())
    axs[1, 1].set_title("Feature Heatmap")
    fig.colorbar(im, ax=axs[1, 1])

    plt.tight_layout()
    
    # If classification mode is enabled, save in the appropriate directory
    # check if the basename is an jpg image
    if args.classify:
        # Extract the class name and correctness from the image path
        path_parts = args.image_path.split(os.sep)
        class_name = path_parts[-3]
        correctness = path_parts[-2] # correct or mistake
        assert correctness in ['correct', 'mistake'], "The image path should contain 'correct' or 'mistake'"

        # Create the full save path under the explanations directory
        save_path = os.path.join('explanations', class_name, correctness, os.path.basename(args.image_path))
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Save the explanation
        plt.savefig(save_path, dpi=300)
        print(f"Explanation saved at {save_path}")
    else:
        # make dir for storing the explanations and save it there with the same name as the image
        os.makedirs("./explanations", exist_ok=True)
        plt.savefig(f"./explanations/{os.path.basename(args.image_path)}")
        print(f"Explanation saved at ./explanations/{os.path.basename(args.image_path)}")
