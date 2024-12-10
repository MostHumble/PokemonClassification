
import torch
import argparse
from utils.inference import preprocess_image, predict
from utils.train_utils import initialize_model

def main():
    parser = argparse.ArgumentParser(description="Image Inference")
    parser.add_argument("--model_name", type=str, required=True, help="Model name (resnet, alexnet, vgg, squeezenet, densenet)", default="resnet")
    parser.add_argument("--model_weights", type=str, required=True, help="Path to the model weights", default="trained_models\pokemon_resnet.pth")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the image", default="pokemonclassification\PokemonData\Abra\2eb2a528f9a247358452b3c740df69a0.jpg")
    parser.add_argument("--num_classes", type=int, required=True, help="Number of classes", default=150)
    args = parser.parse_args()

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

if __name__ == "__main__":
    main()