
ARG PLATFORM=cpu
ARG PETSC_VERSION=3.17.3
ARG SLEPC_VERSION=3.17.1


FROM python:3.9-bullseye AS base-cpu

# install dependencies
# TODO: it would be great to install scalapack here as well, it doesn't seem to work though :-(
ONBUILD RUN apt-get update && \
    \
    DEBIAN_FRONTEND="noninteractive" \
    apt-get install -y --no-install-recommends \
    gfortran \
    cmake \
    mpi-default-dev \
    libopenblas-dev \
    libmumps-dev \
    git \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ARG PETSC_ARCH=complex-opt
ENV PETSC_ARCH=$PETSC_ARCH

ARG PETSC_CONFIG_FLAGS
ENV PETSC_CONFIG_FLAGS="$PETSC_CONFIG_FLAGS --download-scalapack=1 --with-mumps=1"


FROM nvidia/cuda:11.6.0-devel-ubuntu20.04 AS base-gpu

# install dependencies
ONBUILD RUN apt-get update && \
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

ARG PETSC_ARCH=cuda-opt
ENV PETSC_ARCH=$PETSC_ARCH

ARG PETSC_CONFIG_FLAGS
ENV PETSC_CONFIG_FLAGS="$PETSC_CONFIG_FLAGS --with-mpi=0"


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
RUN pip3 install --no-cache-dir --upgrade pip


from build as petsc

ARG PETSC_VERSION

# install PETSc
USER root
WORKDIR /opt
RUN curl --no-progress-meter https://ftp.mcs.anl.gov/pub/petsc/release-snapshots/petsc-$PETSC_VERSION.tar.gz | tar xzf -
RUN mv petsc-$PETSC_VERSION petsc
RUN chown -R dnm:dnm petsc
USER dnm

WORKDIR /opt/petsc
ENV PETSC_DIR=/opt/petsc
COPY --chown=dnm:dnm petsc_config/$PETSC_ARCH.py .
ARG CUDA_ARCH=70
RUN DNM_CUDA_ARCH=$CUDA_ARCH ./$PETSC_ARCH.py ${PETSC_CONFIG_FLAGS} && \
    make all


from build as mpi4py

# mpi4py is useful and required for the tests
# but we don't include it in GPU builds
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

# install python packages
WORKDIR /home/dnm
COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

RUN pip3 install --no-cache-dir $PETSC_DIR/src/binding/petsc4py

COPY --from=slepc /opt/slepc /opt/slepc
ENV SLEPC_DIR=/opt/slepc
RUN pip3 install --no-cache-dir $SLEPC_DIR/src/binding/slepc4py

# install dynamite!
RUN mkdir /home/dnm/dynamite
WORKDIR /home/dnm/dynamite

COPY --chown=dnm:dnm . .

RUN pip3 install ./


FROM python:3.9-slim-bullseye as release-base-cpu

ONBUILD RUN apt-get update && \
    \
    DEBIAN_FRONTEND="noninteractive" \
    apt-get install -y --no-install-recommends \
    mpi-default-bin \
    libopenblas0 \
    libmumps-5.3 \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ARG PETSC_ARCH=complex-opt
ENV PETSC_ARCH=$PETSC_ARCH


FROM nvidia/cuda:11.6.0-runtime-ubuntu20.04 AS release-base-gpu

ONBUILD RUN apt-get update && \
    \
    DEBIAN_FRONTEND="noninteractive" \
    apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    libopenblas0 \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ARG PETSC_ARCH=cuda-opt
ENV PETSC_ARCH=$PETSC_ARCH


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
RUN mkdir -p /home/dnm/work  # for permissions
VOLUME /home/dnm/work
WORKDIR /home/dnm


FROM release AS jupyter

LABEL org.opencontainers.image.authors="Greg Kahanamoku-Meyer <gregory.meyer@gmail.com>"

# install jupyter
RUN pip3 install --no-cache-dir jupyterlab matplotlib

# run it!
WORKDIR /home/dnm
EXPOSE 8887
CMD ["jupyter", "lab", "--no-browser", "--ip=0.0.0.0", "--port=8887"]
