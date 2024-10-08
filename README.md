
# GenAI multimodal demo application for Amazon Bedrock

This project offers a sample multimodal front-end application built with Streamlit to showcase Amazon Bedrock. 
You can deploy this CDK project in your AWS account. Alternatively, you can run the Streamlit application on your local machine. 
See sections below for instructions.


## Architecture
![Architecture](./images/architecture.png)


## Prequisites

Make sure you have access to the Bedrock models before using this application. You can request access to the Bedrock models through the console:

![Model Access](./images/model-access.png)

In this example, we are using the following models:
- Claude 3.5 Sonnet 
- Titan Image Generator G1 v2 
- Stability AI SDXL 1.0


The following tools should be installed on your deployment workstation:
* AWS CLI
* Node.js
* IDE
* AWS CDK Toolkit (v2.0 or later)
* Git, JQ, etc
* Docker


## Deploy application in your AWS account

Clone the repository and enter the project directory:

```
git clone https://github.com/Hantzley/amazon-bedrock-multimodal-demo-app.git
cd amazon-bedrock-multimodal-demo-app

```

Create a virtualenv on MacOS or Linux:

```
python3 -m venv .venv
```

After the init process completes and the virtualenv is created, you can use the following
step to activate your virtualenv.

```
source .venv/bin/activate
```

After the virtual environment is activated, upgrade pip to the latest version:
```
python3 -m pip install --upgrade pip
```

Once the virtualenv is activated, you can install the required dependencies.

```
pip install -r requirements.txt
```

If your account is not yet boostrapped for CDK, execute the following command:

```
cdk bootstrap
```

At this point you can now list the stacks in the project.

```
cdk ls
```
You should see the following output:

```
GenAiBedrockVpcStack
GenAiBedrockWebStack
```

Deploy the application as a container on Elastic Container Services in our AWS account:

```
cdk deploy GenAiBedrockWebStack
```
The `GenAiBedrockWebStack` depends on `GenAiBedrockVpcStack`. CDK will resolve that dependency and automatically deploy `GenAiBedrockVpcStack` first.

Copy the `WebApplicationServiceURL` from the output and paste it on your browser.


## Run application on your local machine

Alternatively, you can execute the Streamlit application on your local machine:

```
cd web-app
streamlit run Home.py 
```

The application should open in your browser.

Enjoy!

## Clean up

To avoid unnecessary cost, you can destroy the resources used in the project:

```
cdk destroy --all
```


## Screenshots

### Text generation

![Text generation](./images/01-text-generation.png)

### Text summarization

![Text summarization](./images/02-text-summarization.png)

### Text to image

![Text to image](./images/03-text-to-image.png)

## Image to image

![Image to image](./images/04-image-to-image.png)
