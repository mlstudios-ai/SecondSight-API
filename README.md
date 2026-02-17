# SecondSight API Endpoint (FastAPI)
This repo is of of the three step components of the <a href="https://github.com/mlstudios-ai/SecondSight">SecondSight</a> project. 

1. SecondSight-MLOps - Model training and machine learning pipeline.
2. SecondSight-API - Model deployment and serving endpoint in FastAPI (this)
3. SecondSight - iOS client app in SwiftUI

Fore more project info, check out <a href="https://github.com/mlstudios-ai/SecondSight">main project</a>  repo.

## Environment Setup

1. Create a new conda environment (or use an existing one):

    ```sh
    conda create -p venv/ python==3.11
    conda activate venv/
    ```

2. Install the required dependencies:

    ```sh
    pip3 install -r api/requirements.txt 
    ```

## Running the Server

To start the server, run the following command:

```sh
uvicorn api.main:app --reload
```

## API endpoint for model inferencing

To test functionalities use {url}/

To test API user {url}/docs
