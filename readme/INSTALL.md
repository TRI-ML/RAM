1. We provide a Docker file to re-create the environment which was used in our experiments under `$RAM_ROOT/docker/Dockerfile`. You can either configure the environment yourself using the docker file as a guide or build it via:
  ~~~
    cd $RAM_ROOT
    make docker-build
    make docker-start-interactive
  ~~~ 

2. The only steps that have to be done manually are installing spatial correlation sampler module and compiling deformable convolutions.

  ~~~
    pip install spatial_correlation_sampler
    
    cd $PermaTrack_ROOT/src/lib/model/networks/
    git clone https://github.com/CharlesShang/DCNv2/ 
    cd DCNv2
    ./make.sh
  ~~~