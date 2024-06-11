import os
import base64
import numpy as np
from PIL import Image
import io
import requests

import replicate
from flask import Flask, request
import gradio as gr
from openai import OpenAI

client = OpenAI()

def image_classifier(moodboard, prompt):

    # Convert the numpy array to a PIL image
    pil_image = Image.fromarray(moodboard.astype('uint8'))
    
    # Save the PIL image to a bytes buffer
    buffered = io.BytesIO()
    pil_image.save(buffered, format="JPEG")
    
    # Encode the image to base64
    image_data = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "You are a product designer. I've attached a moodboard here. In one sentence, what do all of these elements have in common? Answer from a design language perspective, if you were telling another designer to create something similar, including any repeating colors and materials and shapes"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/jpeg;base64," + image_data,
                        },
                    },
                ],
            }
        ],
        max_tokens=300,
    )
    
    openai_response = response.choices[0].message.content
    
    openai_response="test"
    
    # Call Stable Diffusion API with the response from OpenAI
    input = {
        "width": 768,
        "height": 768,
        "prompt": "high quality render of " + prompt + " " + openai_response,
        "negative_prompt": "worst quality, low quality, illustration, 2d, painting, cartoons, sketch",
        "refine": "expert_ensemble_refiner",
        "apply_watermark": False,
        "num_inference_steps": 25
    }
    
    output = replicate.run(
        "stability-ai/sdxl:7762fd07cf82c948538e41f63f77d685e02b063e37e496e96eefd46c929f9bdc",
        input=input
    )
    
    # Download the image from the URL
    image_url = output[0]
    print(image_url)
    response = requests.get(image_url)
    print(response)
    img = Image.open(io.BytesIO(response.content))
    
    return img  # Return the image object


app = Flask(__name__)
os.environ.get("REPLICATE_API_TOKEN")

@app.route("/")
def index():
    demo = gr.Interface(fn=image_classifier, inputs=["image", "text"], outputs="image")
    demo.launch()