
import torch
import argparse
from utils.inference_utils import preprocess_image, predict
from utils.train_utils import initialize_model
from utils.interpretability import lime_interpret_image_inference

def main():
    parser = argparse.ArgumentParser(description="Image Inference")
    parser.add_argument("--model_name", type=str, required=True, help="Model name (resnet, alexnet, vgg, squeezenet, densenet)", default="resnet")
    parser.add_argument("--model_weights", type=str, required=True, help="Path to the model weights", default="trained_models\pokemon_resnet.pth")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the image", default="pokemonclassification\PokemonData\Abra\2eb2a528f9a247358452b3c740df69a0.jpg")
    parser.add_argument("--num_classes", type=int, required=True, help="Number of classes", default=150)
    parser.add_argument("--interpretability", type=bool, required=False, help="Whether to run interpretability or not", default=False)
    args = parser.parse_args()

    if args.interpretability:
        assert args.model_name == "resnet", "Interpretability is only supported for ResNet model for now"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model
    model = initialize_model(args.model_name, args.num_classes)
    model = model.to(device)

    # Load the model weights
    model.load_state_dict(torch.load(args.model_weights))

    # Preprocess the image
    image = preprocess_image(args.image_path, (224, 224))

    # Perform inference
    preds = predict(model, image, device)
    print(f"Predicted class: {preds.item()}")

    if args.interpretability:
        lime_interpret_image_inference(args, model)

if __name__ == "__main__":
    main()