FROM ubuntu:focal
ARG PYTHON_VERSION=3.8
WORKDIR /home/app
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
RUN apt update -y
RUN apt install -y curl
RUN apt install -y gcc g++ --fix-missing

RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda install -y python=$PYTHON_VERSION pillow numpy pyyaml scipy ipython mkl mkl-include ninja cython typing && \
     /opt/conda/bin/conda install pytorch torchvision torchaudio cpuonly -c pytorch && \
     /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
EXPOSE 5000
COPY app.py /home/app/app.py
COPY ml_engine /home/app/ml_engine
COPY templates /home/app/templates
COPY helpers /home/app/helpers
COPY config.py /home/app/config.py
COPY views /home/app/views
CMD ["flask", "run"]
