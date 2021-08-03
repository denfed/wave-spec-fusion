ARG CUDAVERSION=10.2

FROM nvidia/cuda:${CUDAVERSION}-base

# User Setup
ARG UID
ARG GID
ARG USER_PASSWORD
RUN adduser --disabled-password --gecos "" container_user
RUN usermod  -u ${UID} container_user
RUN groupmod -g ${GID} container_user
RUN echo container_user:${USER_PASSWORD} | chpasswd
RUN usermod -aG sudo container_user

RUN mkdir /home/src
WORKDIR /home/src
ENV HOME /home/src

RUN apt update

# Global Apt Dependencies
COPY apt_requirements.txt $HOME/apt_requirements.txt
RUN cat apt_requirements.txt | xargs apt install -y
RUN rm apt_requirements.txt

# Update pip3
RUN pip3 install --upgrade pip==21.1

# Install wandb and initialize
RUN pip3 install --upgrade wandb
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
# RUN wandb login --host=http://172.20.10.25:8080 local-5ed047f6444dbff04bf0737f85516a73125b7b69

# Cache pytorch so it doesn't re-download on requirements change
RUN pip3 install torch==1.8.1

# Global Python Dependencies
COPY pip_requirements.txt $HOME/pip_requirements.txt
RUN pip3 install -r pip_requirements.txt
RUN rm pip_requirements.txt

USER container_user
