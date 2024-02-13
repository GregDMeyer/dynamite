
ARG PLATFORM=cpu
ARG BUILD_TYPE=opt  # 'opt' or 'debug'
ARG PETSC_VERSION=3.20.4
ARG SLEPC_VERSION=3.20.1
ARG CUDA_VERSION=12.2.2
ARG CUDA_UBUNTU_VERSION=22.04
ARG CPU_PYTHON_VERSION=3.11
ARG SCALAR_TYPE=complex


FROM python:${CPU_PYTHON_VERSION} AS base-cpu

# install dependencies
RUN apt-get update && \
    \
    DEBIAN_FRONTEND="noninteractive" \
    apt-get install -y --no-install-recommends \
    gfortran \
    cmake \
    mpi-default-dev \
    libopenblas-dev \
    git \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ARG BUILD_TYPE
ENV PETSC_ARCH=complex-$BUILD_TYPE

ARG SCALAR_TYPE
ARG PETSC_CONFIG_FLAGS
ENV PETSC_CONFIG_FLAGS="$PETSC_CONFIG_FLAGS --download-mumps=1 --with-scalar-type=$SCALAR_TYPE"


FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${CUDA_UBUNTU_VERSION} AS base-gpu

# install dependencies
RUN apt-get update && \
    \
    DEBIAN_FRONTEND="noninteractive" \
    apt-get install -y --no-install-recommends \
    python3-dev \
    python3-venv \
    gfortran \
    cmake \
    curl \
    ca-certificates \
    libopenblas-dev \
    git \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ARG BUILD_TYPE
ENV PETSC_ARCH=cuda-$BUILD_TYPE

ARG CUDA_ARCH=all
ARG SCALAR_TYPE
ARG PETSC_CONFIG_FLAGS
ENV PETSC_CONFIG_FLAGS="$PETSC_CONFIG_FLAGS --with-mpi=0 --with-cuda-arch=$CUDA_ARCH --with-scalar-type=$SCALAR_TYPE"


FROM base-${PLATFORM} as build

# add virtualenvironment to install python packages
RUN python3 -m venv /venv

# switch to non-privileged user
RUN groupadd dnm && useradd --no-log-init -m -g dnm dnm
RUN chown -R dnm:dnm /venv
USER dnm

# activate venv
ENV VIRTUAL_ENV=/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip3 install --no-cache-dir --upgrade pip~=23.0.1 wheel


from build as petsc

ARG PETSC_VERSION
ARG BUILD_TYPE

# install PETSc
USER root
WORKDIR /opt
RUN curl --no-progress-meter https://web.cels.anl.gov/projects/petsc/download/release-snapshots/petsc-$PETSC_VERSION.tar.gz | tar xzf -
RUN mv petsc-$PETSC_VERSION petsc
RUN chown -R dnm:dnm petsc
USER dnm

WORKDIR /opt/petsc
ENV PETSC_DIR=/opt/petsc
COPY --chown=dnm:dnm petsc_config/$PETSC_ARCH.py .
RUN ./$PETSC_ARCH.py ${PETSC_CONFIG_FLAGS} && \
    make all


from build as mpi4py

# mpi4py is not included in GPU builds
ARG PLATFORM
RUN if [ "$PLATFORM" = "cpu" ] ; then pip3 install --no-cache-dir mpi4py ; fi


from petsc as slepc

ARG SLEPC_VERSION

USER root
WORKDIR /opt
RUN curl --no-progress-meter https://slepc.upv.es/download/distrib/slepc-$SLEPC_VERSION.tar.gz | tar xzf -
RUN mv slepc-$SLEPC_VERSION slepc
RUN chown -R dnm:dnm slepc
USER dnm

# install SLEPc
WORKDIR /opt/slepc
ENV SLEPC_DIR=/opt/slepc
RUN ./configure && make


from petsc as dynamite-build

RUN pip3 install --no-cache-dir numpy
RUN pip3 install --no-cache-dir $PETSC_DIR/src/binding/petsc4py

COPY --from=slepc /opt/slepc /opt/slepc
ENV SLEPC_DIR=/opt/slepc

RUN pip3 install --no-cache-dir $SLEPC_DIR/src/binding/slepc4py

RUN mkdir /home/dnm/dynamite
WORKDIR /home/dnm/dynamite

COPY --chown=dnm:dnm . .

RUN pip3 install -v ./


FROM python:${CPU_PYTHON_VERSION}-slim as release-base-cpu

ONBUILD RUN apt-get update && \
    \
    DEBIAN_FRONTEND="noninteractive" \
    apt-get install -y --no-install-recommends \
    mpi-default-bin \
    libopenblas0 \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ARG BUILD_TYPE
ENV PETSC_ARCH=complex-$BUILD_TYPE
ENV OMP_NUM_THREADS=1


FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${CUDA_UBUNTU_VERSION} AS release-base-gpu

RUN apt-get update && \
    \
    DEBIAN_FRONTEND="noninteractive" \
    apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    libopenblas0 \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ARG BUILD_TYPE
ENV PETSC_ARCH=cuda-$BUILD_TYPE
ENV OMP_NUM_THREADS=1


FROM release-base-${PLATFORM} as release

LABEL org.opencontainers.image.authors="Greg Kahanamoku-Meyer <gregory.meyer@gmail.com>"

# get petsc and slepc libraries
COPY --from=petsc /opt/petsc/$PETSC_ARCH/lib /opt/petsc/$PETSC_ARCH/lib
COPY --from=slepc /opt/slepc/$PETSC_ARCH/lib /opt/slepc/$PETSC_ARCH/lib

# switch to non-privileged user
RUN groupadd dnm && useradd --no-log-init -m -g dnm dnm
USER dnm

# get all the python packages
COPY --from=dynamite-build --chown=dnm:dnm /venv /venv
ENV VIRTUAL_ENV=/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# merge in the mpi4py files
COPY --from=mpi4py --chown=dnm:dnm /venv /venv

# include benchmarking and examples
COPY --chown=dnm:dnm benchmarking /home/dnm/benchmarking
COPY --chown=dnm:dnm examples /home/dnm/examples

# so that we can tell we are in a container
ENV DNM_DOCKER=1

# make work directory for mounting local files
# (now deprecated but we leave it for backwards compatibility)
RUN mkdir -p /home/dnm/work  # for permissions
WORKDIR /home/dnm


FROM release AS jupyter

LABEL org.opencontainers.image.authors="Greg Kahanamoku-Meyer <gregory.meyer@gmail.com>"

# install jupyter
RUN pip3 install --no-cache-dir jupyterlab matplotlib

# run it!
WORKDIR /home/dnm
EXPOSE 8887
ENTRYPOINT ["jupyter", "lab", "--no-browser", "--ip=0.0.0.0", "--port=8887"]
