from importlib import reload
from openai import OpenAI
import os
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import base64

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set.")


client = OpenAI()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods, including OPTIONS
    allow_headers=["*"],
)


class PromptRequest(BaseModel):
    prompt: str


class Response(BaseModel):
    response: str


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


@app.post("/prompt")
async def ai_prompt(prompt: PromptRequest):

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant",
            },
            {
                "role": "user",
                "content": prompt.prompt,
            },
        ],  max_tokens=1024)

    return Response(response=response.choices[0].message.content)


@app.post("/uploadfile/")
async def create_upload_file(
    prompt: str = Form(...),
    file: UploadFile = File(None),
):
    base64_image = None
    completion = None
    if file:
        contents = await file.read()
        base64_image = base64.b64encode(contents).decode("utf-8")

        completion = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        },
                    ],
                }
            ],
        )
    else:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],  max_tokens=1024)
    if completion:
        return Response(response=completion.choices[0].message.content)
    return Response(response="No response generated.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("chatapi:app", host="127.0.0.1", port=8000, reload=True)
