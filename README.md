# Simple Retrieval-Augmented Generation (RAG) Application

## Setup

1. Clone the repository:
    ```bash
    git clone git@github.com:volprjir/simple-rag-app.git
    cd simple-rag-app
    ```
2. Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```
   
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Set up OpenAI API key:
    ```bash
    export OPENAI_API_KEY="your-openai-api-key"
    ```

4. Prepare documents and create FAISS index:
    ```bash
    python -m document_preparation.prepare
    ```

5. Run the FastAPI application:
    ```bash
    uvicorn main:app --reload
    ```

## Usage

- Send a POST request to `/generate-response` with the user input:
    ```json
    {
        "user_input": "your query here"
    }
    ```

- The response will contain the generated text based on the retrieved documents and user input.