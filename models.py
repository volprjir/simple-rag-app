from pydantic import BaseModel


class Query(BaseModel):
    user_input: str


class Response(BaseModel):
    generated_text: str
