# Download the file using Invoke-WebRequest
$destination = "./pokemonclassification.zip"
$url = "https://www.kaggle.com/api/v1/datasets/download/lantian773030/pokemonclassification"
Invoke-WebRequest -Uri $url -OutFile $destination

# Extract the zip file to the specified folder
$extractPath = "./pokemonclassification"
Expand-Archive -Path $destination -DestinationPath $extractPath

# Remove the downloaded zip file (if not needed anymore)
Remove-Item $destination
