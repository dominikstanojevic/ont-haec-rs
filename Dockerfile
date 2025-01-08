# Stage: devel
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04 as devel

# Install essential packages
RUN apt-get update \
    && apt-get install -y wget git unzip curl build-essential zlib1g-dev zstd libarchive-tools \
    && rm -rf /var/lib/apt/lists/*

# Build minimap2
RUN git clone https://github.com/lh3/minimap2.git && cd minimap2 && make \
    && cd /

# Download rustc and set PATH
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    && export PATH="/root/.cargo/bin:${PATH}"

# Get libtorch
RUN wget -q -O-p https://download.pytorch.org/libtorch/cu117/libtorch-cxx11-abi-shared-with-deps-2.0.1%2Bcu117.zip |bsdtar -xvf- \
    && export LIBTORCH=/libtorch \
    && export LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH

# Build herro
# Copy files from host to container
RUN git clone https://github.com/lbcb-sci/herro.git herro 

WORKDIR /herro
RUN export LIBTORCH=/libtorch \
    && export LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH \
    && export PATH="/root/.cargo/bin:${PATH}" \
    && RUSTFLAGS="-Ctarget-cpu=native" cargo build -q --release

# Stage: final
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04 as final

# Copy files from the previous stage
COPY --from=devel /minimap2/minimap2 /bin/minimap2
COPY --from=devel /herro/target/release/herro /bin/herro
COPY --from=devel /libtorch /libs/

# Install additional packages
RUN apt-get update \
    && apt-get install -y libgomp1 zlib1g-dev zstd \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV LIBTORCH=/libs
ENV LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH

# Labels
LABEL Author="Dominik Stanojevic"
LABEL Version="v0.0.1"
LABEL Name="herro"
