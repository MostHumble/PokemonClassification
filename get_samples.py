from utils.inference_utils import find_images_from_path
import torch
import argparse
from utils.train_utils import initialize_model


def main():
    parser = argparse.ArgumentParser(description="Image Inference")
    parser.add_argument(
        "--model_name",
        type=str,
        help="Model name (resnet, alexnet, vgg, squeezenet, densenet)",
        default="resnet",
    )
    parser.add_argument(
        "--model_weights",
        type=str,
        help="Path to the model weights",
        default="./trained_models/pokemon_resnet.pth",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        help="Path to the image",
        default="./pokemonclassification/PokemonData/",
    )
    parser.add_argument(
        "--num_classes", type=int,  help="Number of classes", default=150
    )
    parser.add_argument(
        "--label", type=str, help="Label to filter the images", default='Dragonair' # Krabby, Clefairy
        )
    parser.add_argument(
        "--num_correct", type=int, help="Number of correctly classified images", default=5
        )
    parser.add_argument(
        "--num_incorrect", type=int, help="Number of incorrectly classified images", default=5
        )
    
    args = parser.parse_args()

    assert (args.model_name == "resnet"), "Only the ResNet is supported model for now"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model
    model = initialize_model(args.model_name, args.num_classes)
    model = model.to(device)

    # Load the model weights
    model.load_state_dict(torch.load(args.model_weights, map_location=device))
    find_images_from_path(args.image_path, model, device, args.num_correct, args.num_incorrect, args.label)

if __name__ == "__main__":
    main()
