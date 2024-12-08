#!/bin/bash
curl -L -o ./pokemonclassification.zip https://www.kaggle.com/api/v1/datasets/download/lantian773030/pokemonclassification

unzip -d ./pokemonclassification pokemonclassification.zip

# Remove the downloaded zip file (if you don't need it anymore)
rm ~/Downloads/pokemonclassification.zip