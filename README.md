# Hashtag-Generator

### Install

1. Create conda environment
    
    ```python
    conda create -n mplug_owl python=3.10
    conda activate mplug_owl
    ```
    
2. Install PyTorch
    
    ```python
    conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
    ```
    
3. Install dependencies
    
    ```python
    pip install -r requirements.txt
    ```
    
4. Install redis for celery
    
    ```python
    # in linux
    wget http://download.redis.io/redis-stable.tar.gz
    tar xvzf redis-stable.tar.gz
    cd redis-stable
    make
    sudo make install
    redis-server
    ```
    

### Implement

1. celery_worker.py
    
    ```python
    
    celery -A celery_worker worker --loglevel=info --concurrency=1
    # concurrency를 높이면 작업자 수가 늘어남
    # celery 사용하지 않을 시, 아래의 main_cpu 실행 (for cpu)
    ```
    
2. main.py
    
    ```python
    # for cpu 
    uvicorn main_cpu:app --reload --host=0.0.0.0 --port=30000
    # for cpu with celery
    uvicorn main_cpu_celery:app --reload --host=0.0.0.0 --port=30000
    # for cuda with celery
    uvicorn main_cuda_celery:app --reload --host=0.0.0.0 --port=30000 
    ```