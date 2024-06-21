from fastapi import FastAPI, HTTPException

from models import Query, Response
from utils import process_question

app = FastAPI()


@app.post("/generate-response", response_model=Response)
def generate_response(query: Query):
    # try:
    response = process_question(query.user_input)
    return Response(generated_text=response)
    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
