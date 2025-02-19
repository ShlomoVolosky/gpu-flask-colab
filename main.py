from flask import Flask, render_template, request
from flask_ngrok import run_with_ngrok

import torch
from diffusers import StableDiffusionPipelineSafe
from diffusers.pipelines.stable_diffusion_safe import SafetyConfig

import base64
from io import BytesIO

# Load Model
pipeline = StableDiffusionPipelineSafe.from_pretrained(
    "AIML-TUDA/stable-diffusion-safe", torch_dtype=torch.float16
).to("cuda")

# Start Flask app and set to ngrok
app = Flask(__name__)
run_with_ngrok(app)

@app.route('/')
def initial():
    return render_template('index.html')

@app.route('/submit-prompt', methods=['POST'])
def generate_image():
    prompt = request.form["prompt-input"]
    image = pipeline(prompt=prompt, **SafetyConfig.MEDIUM).images[0]

    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue())
    img_str = "data:image/png;base64," + str(img_str)[2:-1]

    return render_template("index.html", generated_image=img_str)

if __name__ == "__main__":
    app.run()
