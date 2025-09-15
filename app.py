# IMPORTS - These are tools from different toolboxes.
# Each import brings in pre-written code for use in the project.
    # Flask = The main web framework (website foundation)
    # render_template = Takes HTML files and shows them to users visiting the site
    # request = Handles data that users send to the app (like form submissions)
    # jsonify = Converts Python data into JSON (web-friendly format JavaScript can read)
from flask import Flask, render_template, request, jsonify
from transformers import pipeline
import torch
import re


# Create the web application
app = Flask(__name__)


# Initialize text generation pipeline (this downloads the model first time)
print("Loading AI model...(this may take a moment on first run)")

try:
    # Code that might fail
    # Text generation using GPT-2
    generator = pipeline(
        # Tells the pipeline what kind of AI task you want
        "text-generation", 
        # Specifies the model to use. We're balancing speed and quality here.
        model="gpt2-medium", 
        # Conditional statement. Checks for a graphics card. 
        device=0 if torch.cuda.is_available() else -1, 
        pad_token_id=50256 # Prevents warnings
    )
    print("✅ AI model loaded successfully!")
except Exception as e:
    # What to do if the code above fails 
    print(f"❌ Error loading model: {e}")
    generator = None


# POEM GENERATION FUNCTION
# Takes the inputs and gives back results as outputs
    # def = "define a function"
    # prompt = required input from user
    # style="free verse" = optional input with a default value
    # length="short" = option input with a default value
def generate_poem(prompt, style="free verse", length="short"):
    """Generate a poem using local GPT-2 model"""
    
    #Safety check: Make sure the AI model actually loads
    if generator is None:
            return "Error: AI model not loaded. Please check your configurations."
        
    try:
        # Create specific instructions for the AI based on user inputs
        if style.lower() == "haiku":
            poetry_prompt = f"Write a haiku poem aboue {prompt}:\n"
            max_length = 50
        elif style.lower() == "limerick"
            poetry_prompt = f"Write a limerick about {prompt}:\n"
            max_length = 80
        elif length == "short":
            poetry_prompt = f"Write a short poem about {prompt}\n"
            max_length = 100
        else:
            poetry_prompt = f"Write a poem about {prompt}\n"
            max_length = 150