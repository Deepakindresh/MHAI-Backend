import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from unsloth import FastLanguageModel
import torch
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Add your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model configuration
max_seq_length = 8000
dtype = None
load_in_4bit = True

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

# Load model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    token="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",
    model_name="Deepakindresh/llama_therapist",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)
FastLanguageModel.for_inference(model)  # Enable native 2x faster inference


# Define request model using Pydantic
class ChatRequest(BaseModel):
    text: str


@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Accepts text input and returns the LLM's response.
    """
    try:
        user_input = request.text

        if not user_input:
            raise HTTPException(status_code=400, detail="No input text provided")

        # Prepare the prompt
        prompt = alpaca_prompt.format(
            "play the role of an empathetic and kind psychotherapist (cognitive-behavioral therapy). You need to use various empathetic strategies, such as understanding, support, and encouragement to give me a more comfortable experience. Be very concise. (Remember: act as a human psychotherapist)",
            user_input,
            "",
        )

        # Tokenize the input
        input_ids = tokenizer([prompt], return_tensors="pt").to("cuda")

        # Generate output
        outputs = model.generate(**input_ids, max_new_tokens=4000, use_cache=True)

        # Decode the response
        response = tokenizer.batch_decode(outputs)[0].split("### Response:")[1].strip()

        return {"response": response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Run the application using Uvicorn
# (Run `uvicorn filename:app --host 0.0.0.0 --port 5000 --reload` to start)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)