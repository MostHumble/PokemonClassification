import torch.nn as nn
from torchvision import transforms 
from utils.data import PokemonDataModule
from utils.train import initialize_model, train_and_evaluate
import torch
import torch.optim as optim


DATA_DIR = "/kaggle/input/pokemonclassification/PokemonData"
INDICIES_FILE = "/kaggle/input/pokindicies/indices.pkl"
IMG_SHAPE = ((224,224))

pokemon_dataset = PokemonDataModule(DATA_DIR)
NUM_CLASSES = len(pokemon_dataset.class_names)

# Get class names
print(f"Number of classes: {NUM_CLASSES}")

# You can only the use precomputed means and vars if using the same indices file ('indices.pkl')
chanel_means = torch.tensor([0.6062, 0.5889, 0.5550])
chanel_vars = torch.tensor([0.3284, 0.3115, 0.3266])
stats = {"mean":chanel_means, "std":chanel_vars}

pokemon_dataset.prepare_data(indices_file=INDICIES_FILE, get_stats=False)

print(f"Train dataset size: {len(pokemon_dataset.train_dataset)}")
print(f"Test dataset size: {len(pokemon_dataset.test_dataset)}")


# Transformations of data for testing
test_transform=transforms.Compose([
        transforms.Resize(IMG_SHAPE),
        transforms.ToTensor(),       # Convert PIL images to tensors
        transforms.Normalize(**stats), # Normalize images using mean and std

])

# Data augmentations for training
train_transform = transforms.Compose([
    transforms.Resize(IMG_SHAPE),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(IMG_SHAPE, padding=4),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(**stats)
])

# get dataloaders
trainloader, testloader= pokemon_dataset.get_dataloaders(train_transform=train_transform,
                                                         test_transform=test_transform,
                                                         train_batch_size=128,
                                                         test_batch_size=512
                                                        )

pokemon_dataset.plot_examples(testloader, stats=stats)

pokemon_dataset.plot_examples(trainloader, stats=stats)



MODELS = ["resnet", "alexnet", "vgg", "squeezenet", "densenet"]

# Try with a finetuning a resnet for example
model = initialize_model(MODELS[0],
                        NUM_CLASSES,
                        feature_extract=True,
                        use_pretrained=True)

# Print the model we just instantiated
print(model)

# Model, criterion, optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Train and evaluate
history = train_and_evaluate(
    model=model,
    trainloader=trainloader,
    testloader=testloader,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    epochs=10
)