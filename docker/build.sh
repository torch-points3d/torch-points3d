#/bin/bash
#
#   Script that builds a docker image containing the code base
#   and a specific set of pretrained weights
#
set -eu

if [[ "$#" -ne 1 ]]
then
  echo "Usage: ./build.sh path/to/kpconv.pt"
  exit 1
fi
MODEL=$1

# Check that file exists
cd ..
if [ ! -f "$MODEL" ]; then
    echo "$MODEL does not exist"
    exit 1
fi

# Sets a bunch of variables
MODEL_NAME="$(basename $MODEL)"

# Add model to docker context (outputs is ignored)
cp $MODEL $MODEL_NAME

# Build image
IMAGE_NAME=$(echo "$MODEL_NAME" | cut -f 1 -d '.'):latest
sudo docker build -f docker/Dockerfile --build-arg MODEL=$MODEL_NAME -t $IMAGE_NAME .

# Cleanup
rm $MODEL_NAME
