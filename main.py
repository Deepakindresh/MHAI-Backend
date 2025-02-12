import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from unsloth import FastLanguageModel
import torch
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import supabase
from supabase.client import Client
from datetime import datetime
import os
from dotenv import load_dotenv
import numpy as np

load_dotenv()

# Initialize FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Supabase client
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
supabase_client: Client = supabase.create_client(supabase_url, supabase_key)

# Model configuration
max_seq_length = 8192
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
# model, tokenizer = FastLanguageModel.from_pretrained(
#     token=os.getenv("HUGGINGFACE_TOKEN"),
#     model_name="Deepakindresh/llama_therapist",
#     max_seq_length=max_seq_length,
#     dtype=dtype,
#     load_in_4bit=load_in_4bit,
# )
# FastLanguageModel.for_inference(model)


# DeepSeek Llama3.1 8B model
deepSeekModel, deepSeekTokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/DeepSeek-R1-Distill-Llama-8B",  # Load the pre-trained DeepSeek R1 model (8B parameter version)
    max_seq_length=max_seq_length, # Ensure the model can process up to 2048 tokens at once
    dtype=dtype, # Use the default data type (e.g., FP16 or BF16 depending on hardware support)
    load_in_4bit=load_in_4bit, # Load the model in 4-bit quantization to save memory
    token=os.getenv("HUGGINGFACE_TOKEN"), # Use hugging face token
)

FastLanguageModel.for_inference(deepSeekModel)

# Define request models
class ChatRequest(BaseModel):
    text: str
    user_id: str
    last4Messages: str
    # modelName: str

class UserProfile(BaseModel):
    full_name: str
    preferred_name: Optional[str]
    age: int
    gender: Optional[str]

class Assessment(BaseModel):
    gad7_score: int
    gad7_severity: str
    phq9_score: int
    phq9_severity: str
    ptsd_positive: bool
    has_suicidal_thoughts: bool
    raw_responses: dict
    assessment_date: datetime

async def get_embeddings(text: str) -> list:
    """Generate embeddings using the model and resize to match database dimensions."""
    try:
        with torch.no_grad():
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2000).to("cuda")
            embeddings = model(**inputs, output_hidden_states=True).hidden_states[-1]
            embeddings = embeddings.float()
            embedding = embeddings.mean(dim=1).squeeze().cpu().numpy()
            
            if len(embedding) != 2000:
                original_embedding = embedding.reshape(1, -1)
                target_size = 2000
                step = len(embedding) // target_size
                resized_embedding = np.zeros(target_size)
                
                for i in range(target_size):
                    start_idx = i * step
                    end_idx = start_idx + step
                    if i == target_size - 1:
                        end_idx = len(embedding)
                    resized_embedding[i] = np.mean(original_embedding[0, start_idx:end_idx])
                
                embedding = resized_embedding

            return embedding.astype(float).tolist()
    except Exception as e:
        print(f"Error generating embeddings: {str(e)}")
        return [0.0] * 2000

async def get_user_profile(user_id: str) -> Optional[UserProfile]:
    """Fetch user profile from database."""
    try:
        response = supabase_client.table('user_profiles') \
            .select('*') \
            .eq('id', user_id) \
            .single() \
            .execute()
        
        return UserProfile(**response.data) if response.data else None
    except Exception as e:
        print(f"Error fetching user profile: {str(e)}")
        return None

async def get_latest_assessment(user_id: str) -> Optional[Assessment]:
    """Fetch user's latest mental health assessment."""
    try:
        response = supabase_client.table('mental_health_assessments') \
            .select('*') \
            .eq('user_id', user_id) \
            .order('assessment_date', desc=True) \
            .limit(1) \
            .execute()
        
        return Assessment(**response.data[0]) if response.data else None
    except Exception as e:
        print(f"Error fetching assessment: {str(e)}")
        return None

async def get_relevant_context(user_id: str, current_message: str) -> str:
    """Get relevant previous conversations."""
    try:
        current_embedding = await get_embeddings(current_message)
        
        response = supabase_client.rpc(
            'match_messages',
            {
                'query_embedding': current_embedding,
                'match_threshold': 0.7,
                'match_count': 4,
                'user_id': user_id
            }
        ).execute()
        
        context_messages = []
        for msg in response.data:
            role = "Bot" if msg['is_bot'] else "User"
            context_messages.append(f"{role}: {msg['content']}")
        
        return "\n".join(context_messages[-4:])
    except Exception as e:
        print(f"Error getting context: {str(e)}")
        return ""


def get_severity_description(assessment: Assessment) -> str:
    """Generate a description of the user's mental health severity using stored values."""
    descriptions = []
    
    # Use stored severity levels directly
    if assessment.gad7_severity != 'Minimal':
        descriptions.append(f"{assessment.gad7_severity.lower()} anxiety")
        
    if assessment.phq9_severity != 'None-minimal':
        descriptions.append(f"{assessment.phq9_severity.lower()} depression")
        
    if assessment.ptsd_positive:
        descriptions.append("PTSD symptoms")
        
    if assessment.has_suicidal_thoughts:
        descriptions.append("suicidal thoughts")
    
    if not descriptions:
        return "minimal symptoms"
        
    return ", ".join(descriptions)


@app.post("/chat")
async def chat(request: ChatRequest):
    """Process chat requests with user context and store in database."""
    try:
        if not request.text:
            raise HTTPException(status_code=400, detail="No input text provided")

        # Get user context
        user_profile = await get_user_profile(request.user_id)
        assessment = await get_latest_assessment(request.user_id)
        context = await get_relevant_context(request.user_id, request.text)

        # Prepare the system context
        system_context = f"""
        User Profile:
        - Name: {user_profile.preferred_name or user_profile.full_name if user_profile else 'User'}
        - Age: {user_profile.age if user_profile else 'Unknown'}
        
        Mental Health Status:
        - Anxiety: {assessment.gad7_severity if assessment else 'Not assessed'} (GAD-7)
        - Depression: {assessment.phq9_severity if assessment else 'Not assessed'} (PHQ-9)
        - PTSD Screening: {'Positive' if assessment and assessment.ptsd_positive else 'Negative' if assessment else 'Not assessed'}
        {'- Has reported suicidal thoughts' if assessment and assessment.has_suicidal_thoughts else ''}
        
        Current Status: {get_severity_description(assessment) if assessment else 'Not assessed'}
        
        Relevant Conversation Context:
        {context}

        Last 4 messages:
        {request.last4Messages}
        """

        # Rest of the chat function remains the same...
        instruction = f"""
        You are a medical expert bot called DandelionAI.
        Play the role of an empathetic and kind psychotherapist (cognitive-behavioral therapy). 
        You are speaking to a {user_profile.age if user_profile else ''} year old person.
        Adjust your language and response appropriately for their age.
        Use various empathetic strategies, such as understanding, support, and encouragement.
        Be concise yet supportive. Remember: You're name is DandelionAI but speak like a human psychotherapist.
        Use the context of previous conversations and relevant conversations and respond to the new input message.
        """

        prompt = alpaca_prompt.format(
            instruction + "\n" + system_context,
            request.text,
            ""
        )

        print(prompt)

        # Generate response
        # if request.modelName == "LLama3":
        # input_ids = tokenizer([prompt], return_tensors="pt").to("cuda")
        # outputs = model.generate(**input_ids, max_new_tokens=4000, use_cache=True)
        # response = tokenizer.batch_decode(outputs)[0].split("### Response:")[1].strip()
        # response = response.replace("<|end_of_text|>", "")

        # DeepSeek Integration but RAM not sufficient as of now. So create separate backend for the other process.
        # elif request.modelName == "DeepSeek":
        input_ids = deepSeekTokenizer([prompt], return_tensors="pt").to("cuda")
        outputs = deepSeekModel.generate(**input_ids, max_new_tokens=4000, use_cache=True)
        response = deepSeekTokenizer.batch_decode(outputs)[0].split("</think>")[1].strip()
        response = response.replace("< | end_of_sentence | >", "")
        print("This is the response: ", response)

        try:
            user_message_embedding = await get_embeddings(request.text)
            bot_message_embedding = await get_embeddings(response)

            # Store both messages in the database
            supabase_client.table('chat_messages').insert([
                {
                    'user_id': request.user_id,
                    'content': request.text,
                    'is_bot': False,
                    'embedding': user_message_embedding
                    # 'modelName': None
                },
                {
                    'user_id': request.user_id,
                    'content': response,
                    'is_bot': True,
                    'embedding': bot_message_embedding
                    # 'modelName': request.modelName
                }
            ]).execute()
        except Exception as e:
            print(f"Error storing messages: {str(e)}")
            pass

        return {
            "response": response,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Run with: uvicorn main:app --host 0.0.0.0 --port 5000 --reload