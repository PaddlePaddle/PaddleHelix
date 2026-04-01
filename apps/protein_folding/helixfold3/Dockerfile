ARG CUDA=12.8.1
FROM nvidia/cuda:${CUDA}-cudnn-devel-ubuntu22.04
# FROM directive resets ARGS, so we specify again (the value is retained if
# previously set).
ARG CUDA

LABEL Author="j.caley@unsw.edu.au" \
    title="helixfold3" \
    description="Docker image containing all software requirements to run helixfold3"

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y wget git && \
    wget -q -P /tmp "https://github.com/conda-forge/miniforge/releases/download/24.11.0-0/Miniforge3-$(uname)-$(uname -m).sh" && \
    bash /tmp/Miniforge3-$(uname)-$(uname -m).sh -b -p /conda && \
    git clone --filter=blob:none --no-checkout https://github.com/PaddlePaddle/PaddleHelix.git /app/helixfold3 && \
    cd /app/helixfold3 && \
    git sparse-checkout init --cone && \
    git sparse-checkout set apps/protein_folding/helixfold3 && \
    git checkout dev && \
    mv apps/protein_folding/helixfold3/* . && \
    rm -rf /tmp/Miniforge3-$(uname)-$(uname -m).sh /var/lib/apt/lists/* /root/.cache apps && \
    apt-get remove --purge -y wget git && apt-get autoremove -y && apt-get clean -y

COPY environment.yaml /app/helixfold3/environment.yaml

ENV PATH="/conda/bin:$PATH"

RUN mamba env create --file=/app/helixfold3/environment.yaml && \
    mamba install -y -c bioconda aria2 hmmer==3.3.2 kalign2==2.04 hhsuite==3.3.0 -n helixfold && \
    mamba install -y -c conda-forge openbabel -n helixfold && \
    mamba clean --all --force-pkgs-dirs -y && \
    rm -rf /root/.cache

ENV PATH="/conda/bin:/app/helixfold3:./maxit_src/bin:/conda/envs/helixfold/bin:$PATH" \
    RCSBROOT="./maxit_src" \
    MAXIT_SRC="./maxit_src" \
    PYTHON_BIN="/conda/envs/helixfold/bin/python3.9" \
    ENV_BIN="/conda/envs/helixfold/bin" \
    OBABEL_BIN="/conda/envs/helixfold/bin"

ENTRYPOINT ["mamba", "run", "--name", "helixfold"]
