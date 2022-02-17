
ARG HARDWARE=cpu
ARG PETSC_VERSION=3.16.4
ARG SLEPC_VERSION=3.16.2


FROM python:3.9-bullseye AS base-cpu

# install dependencies
# TODO: it would be great to install scalapack here as well, it doesn't seem to work though :-(
ONBUILD RUN apt-get update && \
    \
    DEBIAN_FRONTEND="noninteractive" \
    apt-get install -y --no-install-recommends \
    gfortran \
    mpi-default-dev \
    libopenblas-dev \
    libmumps-dev \
    git \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ARG PETSC_ARCH=complex-opt
ENV PETSC_ARCH=$PETSC_ARCH
ENV PETSC_CONFIG_FLAGS=--download-scalapack=1 --with-mumps=1


FROM nvidia/cuda:11.4.1-devel-ubuntu20.04 AS base-gpu

# install dependencies
ONBUILD RUN apt-get update && \
    \
    DEBIAN_FRONTEND="noninteractive" \
    apt-get install -y --no-install-recommends \
    python3-dev \
    python3-pip \
    gfortran \
    curl \
    ca-certificates \
    libopenblas-dev \
    git \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ARG PETSC_ARCH=cuda-opt
ENV PETSC_ARCH=$PETSC_ARCH
ENV PETSC_CONFIG_FLAGS=--with-mpi=0


FROM base-${HARDWARE} as build

LABEL org.opencontainers.image.authors="Greg Meyer <gregory.meyer@gmail.com>"

ARG PETSC_VERSION
ARG SLEPC_VERSION

# install PETSc
WORKDIR /opt
RUN curl --no-progress-meter https://ftp.mcs.anl.gov/pub/petsc/release-snapshots/petsc-$PETSC_VERSION.tar.gz | tar xzf -
WORKDIR /opt/petsc-$PETSC_VERSION
ENV PETSC_DIR=/opt/petsc-$PETSC_VERSION

COPY petsc_config/$PETSC_ARCH.py .
RUN ./$PETSC_ARCH.py ${PETSC_CONFIG_FLAGS} && \
    make all

# install SLEPc
WORKDIR /opt
RUN curl --no-progress-meter https://slepc.upv.es/download/distrib/slepc-$SLEPC_VERSION.tar.gz | tar xzf -
WORKDIR /opt/slepc-$SLEPC_VERSION
ENV SLEPC_DIR=/opt/slepc-$SLEPC_VERSION
RUN ./configure && make

# switch to non-privileged user
RUN groupadd dnm && useradd --no-log-init -m -g dnm dnm
USER dnm
ENV PATH=/home/dnm/.local/bin:${PATH}

# install python packages
RUN pip3 install --user --no-cache-dir --upgrade pip

# they need to be installed in order, so we can't do pip install -r requirements.txt....
COPY requirements.txt requirements.txt
RUN while read PACKAGE; \
        do pip3 install --user --no-cache-dir $PACKAGE; \
    done <requirements.txt

# mpi4py is useful and required for the tests
# but we don't include it in GPU builds
ARG HARDWARE
RUN if [ "$HARDWARE" = "cpu" ] ; then pip3 install --user --no-cache-dir mpi4py ; fi

# install dynamite!

RUN mkdir /home/dnm/dynamite
WORKDIR /home/dnm/dynamite

COPY --chown=dnm:dnm . .

RUN pip3 install --user ./


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


FROM nvidia/cuda:11.4.1-runtime-ubuntu20.04 AS release-base-gpu

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


FROM release-base-${HARDWARE} as release

# get petsc and slepc libraries
ARG PETSC_VERSION
ARG SLEPC_VERSION

COPY --from=build /opt/petsc-$PETSC_VERSION/$PETSC_ARCH/lib /opt/petsc-$PETSC_VERSION/$PETSC_ARCH/lib
COPY --from=build /opt/slepc-$SLEPC_VERSION/$PETSC_ARCH/lib /opt/slepc-$SLEPC_VERSION/$PETSC_ARCH/lib

# switch to non-privileged user
RUN groupadd dnm && useradd --no-log-init -m -g dnm dnm
USER dnm
ENV PATH=/home/dnm/.local/bin:${PATH}

# get all the python packages---it's the only thing in .local/lib
RUN mkdir -p /home/dnm/.local  # for permissions
COPY --from=build --chown=dnm:dnm /home/dnm/.local/bin/pip* /home/dnm/.local/bin/
COPY --from=build --chown=dnm:dnm /home/dnm/.local/lib/ /home/dnm/.local/lib/

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

# install jupyter
RUN pip3 install --user --no-cache-dir jupyterlab matplotlib

# run it!
WORKDIR /home/dnm
EXPOSE 8888
CMD ["jupyter", "lab", "--no-browser", "--ip=0.0.0.0"]
