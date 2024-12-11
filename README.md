# PokemonClassification

This repository explores the training of different models for a vision classification task, with a special focus made on reproducibility, and an attempt to a local interpretability of the decision made by a resnet model using LIME

## Table of Contents

- [PokemonClassification](#pokemonclassification)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Dataset](#dataset)
  - [Training](#training)
  - [Inference](#inference)
  - [Generating Data Samples](#generating-data-samples)
  - [Interpretability](#interpretability)
  - [Contributing](#contributing)

## Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/yourusername/PokemonClassification.git
    cd PokemonClassification
    ```

2. Create a conda environment and activate it:

    ```sh
    conda env create -f environment.yaml
    conda activate pokemonclassification
    ```

## Dataset

To get the data, use the appropriate script based on your operating system:

- On Linux-based systems:
  
    ```shell
    ./get_data.sh
    ```

- On Windows:
  
    ```shell
    ./get_data.ps1
    ```

## Training

To train a model, use the `train.py` script. Here are the parameters you can specify:

```python
def parser_args():
    parser = argparse.ArgumentParser(description="Pokemon Classification")
    parser.add_argument("--data_dir", type=str, default="./pokemonclassification/PokemonData", help="Path to the data directory")
    parser.add_argument("--indices_file", type=str, default="indices_60_32.pkl", help="Path to the indices file")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--train_batch_size", type=int, default=128, help="train Batch size")
    parser.add_argument("--test_batch_size", type=int, default=512, help="test Batch size")
    parser.add_argument("--model", type=str, choices=["resnet", "alexnet", "vgg", "squeezenet", "densenet"], default="resnet", help="Model to be used")
    parser.add_argument("--feature_extract", type=bool, default=True, help="whether to freeze the backbone or not")
    parser.add_argument("--use_pretrained", type=bool, default=True, help="whether to use pretrained model or not")
    parser.add_argument("--experiment_id", type=int, default=0, help="Experiment ID to log the results")
    return parser.parse_args()
```

Example:

```shell
python train.py--model resnet --data_dir data/PokemonData --epochs 10 --train_batch_size 32 --test_batch_size 32
```

## Inference

To perform inference on a single image, use the `inference.py` script. Here are the parameters you can specify:

```python
def main():
    parser = argparse.ArgumentParser(description="Image Inference")
    parser.add_argument("--model_name", type=str, help="Model name (resnet, alexnet, vgg, squeezenet, densenet)", default="resnet")
    parser.add_argument("--model_weights", type=str, help="Path to the model weights", default="./trained_models/pokemon_resnet.pth")
    parser.add_argument("--image_path", type=str, help="Path to the image", default="./pokemonclassification/PokemonData/Chansey/57ccf27cba024fac9531baa9f619ec62.jpg")
    parser.add_argument("--num_classes", type=int, help="Number of classes", default=150)
    parser.add_argument("--lime_interpretability", action="store_true", help="Whether to run interpretability or not")
    parser.add_argument("--classify", action="store_true", help="Whether to classify the image when saving the lime filter")
    args = parser.parse_args()

    if args.lime_interpretability:
        assert args.model_name == "resnet", "Interpretability is only supported for ResNet model for now"
```

Example:

```shell
python inference.py --model_name resnet --model_weights path_to_your_model_weights.pth --image_path path_to_your_image.jpg --num_classes 10
```

## Generating Data Samples

To generate data samples, use the `get_samples.py` script. Here are the parameters you can specify:

```python
def main():
    parser = argparse.ArgumentParser(description="Generate Data Samples")
    parser.add_argument("--model_name", type=str, help="Model name (resnet, alexnet, vgg, squeezenet, densenet)", default="resnet")
    parser.add_argument("--model_weights", type=str, help="Path to the model weights", default="./trained_models/pokemon_resnet.pth")
    parser.add_argument("--image_path", type=str, help="Path to the image", default="./pokemonclassification/PokemonData/")
    parser.add_argument("--num_classes", type=int, help="Number of classes", default=150)
    parser.add_argument("--label", type=str, help="Label to filter the images", default='Dragonair')
    parser.add_argument("--num_correct", type=int, help="Number of correctly classified images", default=5)
    parser.add_argument("--num_incorrect", type=int, help="Number of incorrectly classified images", default=5)
    args = parser.parse_args()
```

Example:

```shell
python get_samples.py --model_name resnet --model_weights path_to_your_model_weights.pth --image_path path_to_your_image_directory --num_classes 10 --label Pikachu --num_correct 5 --num_incorrect 5
```

## Interpretability

To interpret the model's predictions using LIME, use the `inference.py` script with the `--lime_interpretability` flag.

Example:

```shell
python inference.py --model_name resnet --model_weights path_to_your_model_weights.pth --image_path path_to_your_image.jpg --num_classes 10 --lime_interpretability
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.
