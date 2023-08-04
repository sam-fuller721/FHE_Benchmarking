FROM ubuntu:20.04 
ENV DEBIAN_FRONTEND noninteractive

# get minimum python version for pyfhel 
RUN apt-get update -y 
RUN apt-get install -y software-properties-common 
RUN add-apt-repository -y ppa:deadsnakes/ppa 
RUN apt-get install -y python3.9 
RUN apt-get install -y python3-pip  

# install python dev drivers and git  
RUN apt-get install -y python3.9-dev 
RUN apt-get install -y git

# install pyfhel 
RUN python3.9 -m pip install pyfhel 
