FROM gitpod/workspace-full:latest

USER gitpod

# Install Redis.
RUN sudo apt-get update \
    && sudo apt-get install -y \
    redis-server \
    && sudo rm -rf /var/lib/apt/lists/*

RUN pip3 install \
    celery==5.2.2 \
    dotmap==1.3.30 \
    fastapi==0.75.2 \
    label-studio==1.4.1.post1 \
    matplotlib==3.5.2 \
    metaflow==2.6.0 \
    numpy==1.22.3 \
    opencv-python==4.5.5.64 \
    pandas==1.3.5 \
    plotly==5.7.0 \
    Pillow==9.0.0 \
    pytest==7.1.2 \
    pytorch-lightning==1.6.3 \
    scikit-image==0.19.2 \
    scikit-learn==1.0.2 \
    scipy==1.8.0 \
    torch==1.11.0 \
    torchmetrics==0.8.1 \
    torchvision==0.12.0 \
    tqdm==4.64.0 \
    uvicorn[standard]==0.17.6 \
    wandb==0.12.16
