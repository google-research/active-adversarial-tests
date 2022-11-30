docker run -it --rm --gpus all --ipc=host --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --shm-size=32g \
  -v ~/projects:/projects \
  nvcr.io/nvidia/tensorflow:21.10-tf1-py3 /bin/bash