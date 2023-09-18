FROM gitpod/workspace-full:latest

USER gitpod

# Install Redis.
RUN sudo apt-get update \
    && sudo apt-get install -y \
    redis-server \
    && sudo rm -rf /var/lib/apt/lists/*

# Install python version
RUN pyenv install 3.8.13
RUN pyenv local 3.8.13

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt