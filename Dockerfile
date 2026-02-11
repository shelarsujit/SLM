# Start from NVIDIA's optimized PyTorch container
# Includes CUDA 12.3, cuDNN 9, and optimized PyTorch
FROM nvcr.io/nvidia/pytorch:24.02-py3

# Set environment variables to non-interactive
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Install core python dependencies
# Pinning versions is critical for reproducibility
RUN pip install --no-cache-dir \
    transformers==4.39.0 \
    accelerate==0.28.0 \
    datasets==2.18.0 \
    wandb==0.16.4 \
    sentencepiece==0.2.0 \
    ninja \
    packaging \
    bitsandbytes==0.43.0

# Install FlashAttention-2
# We build from source to ensure it links against the container's CUDA
RUN pip install flash-attn --no-build-isolation

# Set the working directory
WORKDIR /workspace

# Copy the local code into the container
COPY. /workspace

# Default entry command
CMD ["/bin/bash"]