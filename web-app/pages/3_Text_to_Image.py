import streamlit as st
import boto3
import base64
from io import BytesIO
from PIL import Image
import json
import random
import os

st.set_page_config(
    page_title="Text to Image",
    layout="wide",
)

c1, c2 = st.columns([1, 8])
with c1:
    st.image("./imgs/bedrock.png", width=100)

with c2:
    st.header("Text to Image generation")
    st.caption("Using Stability AI and Amazon Titan image models in Amazon Bedrock")

# Initialize the Bedrock client
region_name = os.getenv('AWS_REGION', 'us-east-1')
boto3_bedrock = boto3.client(service_name='bedrock-runtime', region_name=region_name)

def generate_image_titan(prompt, cfg_scale, seed, image_size):
    dimensions = image_size.split("x")
    body = {
        "taskType": "TEXT_IMAGE",
        "textToImageParams": {
            "text": prompt
        },
        "imageGenerationConfig": {
            "numberOfImages": 1,
            "quality": "standard",
            "cfgScale": cfg_scale,
            "seed": seed,
            "width": int(dimensions[0]),
            "height": int(dimensions[1]),
        }
    }
    
    response = boto3_bedrock.invoke_model(
        body=json.dumps(body),
        modelId="amazon.titan-image-generator-v2:0",
        contentType="application/json",
        accept="application/json"
    )
    
    response_body = json.loads(response['body'].read())
    image_data = base64.b64decode(response_body['images'][0])
    return Image.open(BytesIO(image_data))

def generate_image_stability(prompt, cfg_scale, seed, steps, image_size):
    dimensions = image_size.split("x")
    body = {
        "text_prompts": [{"text": prompt}],
        "cfg_scale": cfg_scale,
        "seed": seed,
        "steps": steps,
        "width": int(dimensions[0]),
        "height": int(dimensions[1]),
    }
    
    response = boto3_bedrock.invoke_model(
        body=json.dumps(body),
        modelId="stability.stable-diffusion-xl-v1",
        contentType="application/json",
        accept="application/json"
    )
    
    response_body = json.loads(response['body'].read())
    image_data = base64.b64decode(response_body['artifacts'][0]['base64'])
    return Image.open(BytesIO(image_data))

prompt = st.text_area("Enter your prompt:","Create an alien planet’s ecosystem, featuring bizarre flora and fauna, strange geology, and atmospheric anomalies.")
with st.sidebar:
    model = st.selectbox("Select Image Model", ["Amazon Titan Model", "Stability AI Model"])
    
    if model == "Amazon Titan Model":
        image_size = st.selectbox("Image Size",["1024x1024", "768x768", "512x512", "768x1152", 
                                                "384x576", "1152x768", "576x384", "768x1280", 
                                                "384x640", "448x576", "1152x896", "576x448",
                                                "768x1408", "384x704", "1408x768", "704x384",
                                                "640x1408", "320x704", "1408x640", "704x320",
                                                "1152x640", "1173x640"],5)
        cfg_scale = st.slider("CFG Scale", 1.1, 10.0, 8.0, 0.1)
    else:  # Stability AI
        image_size = st.selectbox("Image Size",["1024x1024", "1152x896", "1216x832", "1344x768", 
                                                "1536x640", "640x1536", "768x1344", "832x1216", 
                                                "896x1152"],1)
        cfg_scale = st.slider("CFG Scale", 0, 35, 7, 1)
        steps = st.slider("Steps", 10, 50, 30)

if st.button("Generate Image"):
    if prompt:
        with st.spinner("Generating image..."):
            try:
                if model == "Amazon Titan Model":
                    image = generate_image_titan(prompt, cfg_scale, random.randint(0, 2147483646), image_size)
                else:  # Stability AI
                    image = generate_image_stability(prompt, cfg_scale, random.randint(0, 4294967295), steps, image_size)
                
                st.image(image, caption="Generated Image", use_column_width="auto")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter a prompt before generating an image.")