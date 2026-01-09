from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class HelloResponse(BaseModel):
    output: str

@app.get("/hello", response_model=HelloResponse)
def hello():
    return {"output": "hello hottie"}