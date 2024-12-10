from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from inference_utils import predict
import os
import torch 

def unnormalize(image):
    # Unnormalize the image
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=image.dtype)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=image.dtype)
    image = image * torch.tensor(std)
    image = image + torch.tensor(mean)
    image = image.permute(1, 2, 0)
    return image


def lime_interpret_image_inference(args, model, image):
    # Initialize LIME
    explainer = lime_image.LimeImageExplainer()
    # Path to the image you want to explain
    # define a partial function to pass to lime that takes the model as the first parameter
    def predict_fn(x):
        return predict(model, x, args.device)
    # Explain the image
    explanation = explainer.explain_instance(image, predict_fn, top_labels=5, hide_color=0, num_samples=1000)
    # Get the explanation for the top class
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=10, hide_rest=False)
    img_boundry1 = mark_boundaries(temp / 255.0, mask)

    img_boundry1 = unnormalize(img_boundry1)
    # make dir for storing the explanations and save it there with the same name as the image
    os.makedirs("./explanations", exist_ok=True)
    plt.imsave(f"./explanations/{os.path.basename(args.image_path)}", img_boundry1)
    print(f"Explanation saved at ./explanations/{os.path.basename(args.image_path)}")

    # Display the image with explanations
    plt.imshow(img_boundry1)
    plt.title('LIME Explanation')
    plt.show()