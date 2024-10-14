FROM kaggle/python

RUN apt-get update && apt-get install -y \
    htop \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY . /workspace

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh \
    && bash miniconda.sh -b -p /opt/conda \
    && rm miniconda.sh

ENV PATH /opt/conda/bin:$PATH

RUN conda create -n rl-env python=3.8 -y \
    && echo "source activate rl-env" > ~/.bashrc

RUN conda install -n rl-env \
    tensorflow \
    pandas \
    numpy \
    jupyter \
    neptune \
    keras \
    && conda clean --all -y

RUN /opt/conda/envs/rl-env/bin/pip install neptune-client

EXPOSE 8888

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

