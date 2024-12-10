import torch.nn as nn
from torchvision import transforms
from utils.data import PokemonDataModule
from utils.train import initialize_model, train_and_evaluate
import torch
import torch.optim as optim
import mlflow
import argparse
import random

# The shape of the images that the models expects
IMG_SHAPE = (224, 224)


def parser_args():
    parser = argparse.ArgumentParser(description="Pokemon Classification")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./pokemonclassification/PokemonData",
        help="Path to the data directory",
    )
    parser.add_argument(
        "--indices_file",
        type=str,
        default="indices_60_32.pkl",
        help="Path to the indices file",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--train_batch_size", type=int, default=128, help="train Batch size"
    )
    parser.add_argument(
        "--test_batch_size", type=int, default=512, help="test Batch size"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["resnet", "alexnet", "vgg", "squeezenet", "densenet"],
        default="resnet",
        help="Model to be used",
    )
    parser.add_argument(
        "--feature_extract",
        type=bool,
        default=True,
        help="whether to freeze the backbone or not",
    )
    parser.add_argument(
        "--use_pretrained",
        type=bool,
        default=True,
        help="whether to use pretrained model or not",
    )
    parser.add_argument(
        "--experiment_id",
        type=int,
        default=0,
        help="Experiment ID to log the results",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parser_args()

    pokemon_dataset = PokemonDataModule(args.data_dir)
    NUM_CLASSES = len(pokemon_dataset.class_names)

    # Get class names
    print(f"Number of classes: {NUM_CLASSES}")

    # You can only the use precomputed means and vars if using the same indices file ('indices_60_32.pkl')
    if "indices_60_32.pkl" in args.indices_file:
        chanel_means = torch.tensor([0.6062, 0.5889, 0.5550])
        chanel_vars = torch.tensor([0.3284, 0.3115, 0.3266])
        stats = {"mean": chanel_means, "std": chanel_vars}
        _ = pokemon_dataset.prepare_data(
            indices_file=args.indices_file, get_stats=False
        )
    else:
        stats = pokemon_dataset.prepare_data(
            indices_file=args.indices_file, get_stats=True
        )

    print(f"Train dataset size: {len(pokemon_dataset.train_dataset)}")
    print(f"Test dataset size: {len(pokemon_dataset.test_dataset)}")

    # Transformations of data for testing
    test_transform = transforms.Compose(
        [
            transforms.Resize(IMG_SHAPE),
            transforms.ToTensor(),  # Convert PIL images to tensors
            transforms.Normalize(**stats),  # Normalize images using mean and std
        ]
    )

    # Data augmentations for training
    train_transform = transforms.Compose(
        [
            transforms.Resize(IMG_SHAPE),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(IMG_SHAPE, padding=4),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(**stats),
        ]
    )

    # get dataloaders
    trainloader, testloader = pokemon_dataset.get_dataloaders(
        train_transform=train_transform,
        test_transform=test_transform,
        train_batch_size=args.train_batch_size,
        test_batch_size=args.test_batch_size,
    )

    pokemon_dataset.plot_examples(testloader, stats=stats)

    pokemon_dataset.plot_examples(trainloader, stats=stats)

    # Try with a finetuning a resnet for example
    model = initialize_model(
        args.model,
        NUM_CLASSES,
        feature_extract=args.feature_extract,
        use_pretrained=args.use_pretrained,
    )

    # Print the model we just instantiated
    print(model)

    # Model, criterion, optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with mlflow.start_run(
        experiment_id=args.experiment_id,
        run_name=f"{args.model}_{'finetuning' if not args.feature_extract else 'feature_extracting'}"
        f"_{'pretrained' if args.use_pretrained else 'not_pretrained'}"
        f"_{args.indices_file}_{random.randint(0, 1000)}",
    ) as run:
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("lr", args.lr)
        mlflow.log_param("train_batch_size", args.train_batch_size)
        mlflow.log_param("test_batch_size", args.test_batch_size)
        mlflow.log_param("model", args.model)
        mlflow.log_param("feature_extract", args.feature_extract)
        mlflow.log_param("use_pretrained", args.use_pretrained)

        # Train and evaluate
        history = train_and_evaluate(
            model=model,
            trainloader=trainloader,
            testloader=testloader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epochs=args.epochs,
            use_mlflow=True,
        )
        # Save the model
        torch.save(model.state_dict(), f"pokemon_{args.model}.pth")
        mlflow.log_artifact(f"pokemon_{args.model}.pth")
        mlflow.end_run()
