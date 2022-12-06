# Must use a Cuda version 11+
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

WORKDIR /

# pytorch
RUN conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

# Install git
RUN apt-get update && apt-get install -y git 

# Install python packages
RUN pip install --upgrade pip
ADD requirements.txt requirements.txt
RUN pip install -r requirements.txt

# We add the banana boilerplate here
ADD server.py .

# Add your model weight files 
ADD download.py .
RUN python3 download.py

# Add your custom app code, init() and inference()
ADD app.py .

EXPOSE 8000

CMD python3 -u server.py
