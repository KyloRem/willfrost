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
        # Specifies the model to use. Balance speed and quality
        model="gpt2-medium", 
        # Conditional statement. Checks for a graphics card 
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

        # ASKING THE AI MODEL TO GENERATE TEXT
        result = generator(
            poetry_prompt,
            # ^ This is the instruction we created in the generate_poem function
            # The model will read this and continue writing from where it ends
            max_length=len(poetry_prompt.split()) + max_length,
            # ^ len(poetry_prompt.split()) = count words in the prompt
            # + max_length = add the poem length we want
            # Example: if prompt is 6 words + we want 50 more = total 56 words maximum
            num_return_sequences=1,
            # ^ We only want one poem as output, not multiple.
            # This tells the model how many to generate.
            temperature=0.8,
            # ^ Scale of creativity from 0.0 to 1.0
            # 0.0 = very predictable, boring (always picks most likely words)
            # 1.0 = very random, may not make sense
            # 0.8 = creative but coherent
            do_sample=True,
            # ^ True = be creative, try different word combos
            # False = always pick the most statistically likely next word (boring)
            pad_token_id=generator.tokenizer.eos_token_id
            # ^ This tells the model how to handle the end of text
            # Just prevents warnings
        )
        # EXTRACT THE POEM FROM THE MODEL'S RESPONSE
        generated_text = result[0]['generated_text']
        # ^ The model returns a list with one dictionary
        # result[0] = get the first (and only) results
        # ['generated_text'] = get the actual text the model wrote
        poem = generated_text[len(poetry_prompt):].strip()
        # ^ Removes the original prompt, keeps just the poem
        # len(poetry_prompt) = length of our prompt ("Write a haiku about pancakes:")
        # [len(poetry_prompt):] = everythin AFTER the prompt
        # .strip() = Remove extra spaces and newlines from beginning and end

        # CLEAN UP THE AI'S OUTPUT
        # Sometimes models generate messy text, so we clean it up
        lines = poem.split('\n')  
        # ^ Split the poem into individual lines
        # Example: "Line 1\nLine 2\nLine 3" becomes ["Line 1", "Line 2", "Line 3"]
        
        clean_lines = []  
        # ^ Start with an empty list to store good lines
        
        for line in lines:
            # ^ Go through each line one by one
            line = line.strip()  
            # ^ Remove extra spaces from this line
            
            # CHECK IF THIS LINE IS GOOD ENOUGH TO KEEP
            if line and not line.startswith('[') and len(line) > 3:
                # ^ line = make sure line isn't empty
                # ^ not line.startswith('[') = skip lines starting with brackets (AI sometimes adds [INST] tags)
                # ^ len(line) > 3 = skip very short lines (probably noise)
                
                clean_lines.append(line)  
                # ^ Add this good line to our collection
                
                # STOP WHEN WE HAVE ENOUGH LINES for the poem style
                if style.lower() == "haiku" and len(clean_lines) >= 3:
                    break  # Haikus have 3 lines, so stop here
                elif style.lower() == "limerick" and len(clean_lines) >= 5:
                    break  # Limericks have 5 lines, so stop here
                elif len(clean_lines) >= 8:
                    break  # For other poems, stop at 8 lines maximum
          
        # PUT THE CLEAN LINES BACK TOGETHER
        final_poem = '\n'.join(clean_lines[:8])  
        # ^ Join our good lines back together with line breaks
        # clean_lines[:8] = take maximum 8 lines (safety limit)
        # '\n'.join() = put newlines between each line
        
        # BACKUP PLAN: If cleaning left us with nothing, provide a fallback poem
        if not final_poem:
            final_poem = f"A poem about {prompt} flows through my mind,\nLike whispers of thoughts intertwined."
            # ^ If something went wrong and we have no poem, create a simple backup
            # Uses the user's prompt in a generic but nice fallback
            
        return final_poem  
        # ^ Send the finished poem back to whoever called this function
        
    except Exception as e:
        # ^ If anything goes wrong during poem generation, we end up here
        return f"Error generating poem: {str(e)}"
        # ^ Return error message instead of crashing the entire program
