from flask import Flask, render_template, request, jsonify
from transformers import pipeline
import torch
import re

# Create the web application
app = Flask(__name__)

#Initialize text generation pipeline (this downloads the model first time)
print("Loading AI model...(this may take a moment on first run)")

try:
    #Code that might fail
    # Text generation using GPT-2
    generator = pipeline(
        "text-generation", #Tells the pipeline what kind of AI task you want
        model="gpt2-medium", #Specifies the model to use. We're balancing speed and quality here.
        device=0 if torch.cuda.is_available() else -1, #Conditional statement. Checks for a graphics card, use it if so. If not, use regular processer. Slower, but works on any computer.
        pad_token_id=50256 #Prevents warnings
    )
    print("✅ AI model loaded successfully!")
except Exception as e:
    #What to do if the code above fails 
    print(f"❌ Error loading model: {e}")
    generator = None

