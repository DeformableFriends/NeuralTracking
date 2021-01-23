FROM nvcr.io/nvidia/pytorch:20.03-py3 AS main

## Install dependencies
RUN apt-get update
RUN apt-get install -y xorg-dev \
		libglu1-mesa-dev \
		python3-dev \
		libsdl2-dev \ 
		libc++-7-dev \ 
		libc++abi-7-dev \ 
		ninja-build \ 
		libxi-dev \ 
        libtbb-dev \
		libosmesa6-dev \
        libusb-1.0-0-dev \
		build-essential  \ 
		manpages-dev

## Support X11 forwarding in docker
# replace with other Ubuntu version if desired
# see: https://hub.docker.com/r/nvidia/opengl/
COPY --from=nvidia/opengl:1.0-glvnd-runtime-ubuntu16.04 \
  /usr/local/lib/x86_64-linux-gnu \
  /usr/local/lib/x86_64-linux-gnu

# replace with other Ubuntu version if desired
# see: https://hub.docker.com/r/nvidia/opengl/
COPY --from=nvidia/opengl:1.0-glvnd-runtime-ubuntu16.04 \
  /usr/local/share/glvnd/egl_vendor.d/10_nvidia.json \
  /usr/local/share/glvnd/egl_vendor.d/10_nvidia.json

RUN echo '/usr/local/lib/x86_64-linux-gnu' >> /etc/ld.so.conf.d/glvnd.conf && \
    ldconfig && \
    echo '/usr/local/$LIB/libGL.so.1' >> /etc/ld.so.preload && \
    echo '/usr/local/$LIB/libEGL.so.1' >> /etc/ld.so.preload

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics
	
#Install python dependencies & NNRT/csrc
COPY docker_req/req.txt /workspace/reqs/
RUN pip install -r /workspace/reqs/req.txt 
RUN pip install torch==1.7.0+cu110 torchvision==0.8.1+cu110 torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html

COPY . /workspace/NNRT/
WORKDIR /workspace/NNRT/csrc
RUN python setup.py install

RUN mkdir /workspace/local_station
RUN pip install tensorboardX
RUN pip install flowiz -U

EXPOSE 8097
EXPOSE 6006
EXPOSE 8888
WORKDIR /workspace
