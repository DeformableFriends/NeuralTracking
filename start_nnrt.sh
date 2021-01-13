#!/bin/sh
# Varibles to edit - NNRT Repo absolute path
LOCAL_SRC_DIR=/home/eposner/Repositories/NeuralTracking


xhost +LOCAL:

#Build
docker stop portainer
docker stop ml_docker_ne
docker rm portainer
docker rm ml_docker_ne
docker build -t ml_docker_ne .  --build-arg NNRT_PATH=$LOCAL_SRC_DIR

#Running
docker volume create portainer_dat
if [ ! "$(docker ps -q -f name=portainer)" ]; then
    docker stop portainer
    if [ "$(docker ps -aq -f status=exited -f name=portainer)" ]; then
        # cleanup
        docker rm portainer
    fi
    # run your container
    docker run -d -p 8001:8000 -p 9001:9000 --name=portainer --restart=always -v /var/run/docker.sock:/var/run/docker.sock -v portainer_data:/data portainer/portainer
fi 

docker run --rm -p 8888:8888  -p 8097:8097 -p 6006:6006 \
-e DISPLAY=$DISPLAY \
-v /tmp/.X11-unix:/tmp/.X11-unix:rw  \
--privileged \
-e DISPLAY=unix$DISPLAY \
--device /dev/dri  \
-v $LOCAL_SRC_DIR:/workspace/local_station \
--name ml_docker_ne \
--gpus all \
--shm-size=8g \
--ulimit memlock=-1 \
-it ml_docker_ne
 

