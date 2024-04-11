import requests
import onnxruntime
import json
from transformers import BertTokenizer, BertConfig
import numpy as np

# URL of the ONNX file
url = "https://weightsforemb.s3.us-east-1.amazonaws.com/model_optimized.onnx"

# Send GET request to download the ONNX file
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    # Save the downloaded ONNX file
    with open("model_optimized.onnx", "wb") as f:
        f.write(response.content)
    print("ONNX file downloaded successfully.")

    # Load ONNX model
    ort_session = onnxruntime.InferenceSession("model_optimized.onnx")

    # Load other necessary files
    config_path = "config.json"
    ort_config_path = "ort_config.json"
    special_tokens_path = "special_tokens_map.json"
    tokenizer_path = "tokenizer.json"
    tokenizer_config_path = "tokenizer_config.json"
    vocab_path = "vocab.txt"

    # Load config
    with open(config_path, "r") as f:
        config = json.load(f)

    # Load ORT config
    with open(ort_config_path, "r") as f:
        ort_config = json.load(f)

    # Load special tokens
    with open(special_tokens_path, "r") as f:
        special_tokens = json.load(f)

    # Load tokenizer
    tokenizer = BertTokenizer(tokenizer_file=tokenizer_path, vocab_file=vocab_path)
    tokenizer_config = BertConfig.from_pretrained(tokenizer_config_path)
    tokenizer_config.vocab_file = vocab_path


    async def get_text_embeddings(text):
        # Tokenize the input text
        inputs = tokenizer(text, return_tensors="np", padding=True, truncation=True)
        # Prepare inputs for ONNX runtime
        ort_inputs = {input_meta.name: np.atleast_2d(inputs[input_meta.name]) for input_meta in ort_session.get_inputs()}
        # Run inference
        outputs = ort_session.run(None, ort_inputs)
        # Extract embeddings for the [CLS] token (or the first token)
        token_embeddings = outputs[0]
        token_embeddings = np.mean(token_embeddings, axis=1)
        return token_embeddings.reshape((-1,)).astype(np.float16)

else:
    print("Failed to download ONNX file:", response.status_code)
