import streamlit as st
import json
import time
import boto3
import os


st.set_page_config(
    page_title="Text Summarization",
    layout="wide",
)
c1, c2 = st.columns([1, 8])
with c1:
    st.image("./imgs/bedrock.png", width=100)

with c2:
    st.header("Text generation")
    st.caption("Using Claude Sonnet 3.5 model in Amazon Bedrock")


def generate_message(bedrock_runtime, model_id, system_prompt, messages, max_tokens, temperature, top_p, top_k):

    body=json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "system": system_prompt,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
        }  
    )
    
    response = bedrock_runtime.invoke_model(body=body, modelId=model_id)
    response_body = json.loads(response.get('body').read())
   
    return response_body


region_name = os.getenv('AWS_REGION', 'us-east-1')
boto3_bedrock = boto3.client(service_name='bedrock-runtime', region_name=region_name)

modelId = 'anthropic.claude-3-5-sonnet-20240620-v1:0' # change this to use a different version from the model provider
accept = 'application/json'
contentType = 'application/json'
system_prompt = "You are Claude, an AI assistant created by Anthropic to be helpful, \
                harmless, and honest. Your goal is to provide informative and substantive responses \
                to queries while avoiding potential harms."

sample_instruction = """Write an email from Bob, Customer Service Manager, to the customer "John Doe" that provided negative feedback on the service provided by our customer support engineer."""


instruction = st.text_area("Prompt:", sample_instruction, height=100)


max_tokens_to_sample = st.sidebar.slider("max_tokens_to_sample:", min_value=500, max_value=4096, value=4096)
temperature = st.sidebar.slider("temperature:", min_value=0.0, max_value=1.0, value=0.8, step=0.1)
top_k = st.sidebar.slider('top_k:', min_value=10, max_value=500, value=250, step=10)
top_p = st.sidebar.slider('top_p:', min_value=0.0, max_value=1.0, value=0.5, step=0.1)


if st.button("Generate Response", key=instruction):
    if instruction == "":        
        st.error("Please enter a prompt...")
    else:
        with st.spinner("Wait for it..."):    
            
            start_time = time.time()

            user_message =  {"role": "user", "content": instruction}
            messages = [user_message]

            response = generate_message (boto3_bedrock, modelId, system_prompt, 
                                         messages, max_tokens_to_sample,
                                         temperature, top_p, top_k)
            
            st.write(response['content'][0]['text'])

            execution_time = round(time.time() - start_time, 2)

            st.success("Done!")
            st.caption(f"Execution time: {execution_time} seconds")