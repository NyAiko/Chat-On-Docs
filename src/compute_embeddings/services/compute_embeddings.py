import onnxruntime
import json
from transformers import BertTokenizer, BertConfig
import numpy as np
import os

# Define paths relative to the current script file
model_path = os.path.join(os.path.dirname(__file__), "model_optimized.onnx")
config_path = os.path.join(os.path.dirname(__file__), "config.json")
ort_config_path = os.path.join(os.path.dirname(__file__), "ort_config.json")
special_tokens_path = os.path.join(os.path.dirname(__file__), "special_tokens_map.json")
tokenizer_path = os.path.join(os.path.dirname(__file__), "tokenizer.json")
tokenizer_config_path = os.path.join(os.path.dirname(__file__), "tokenizer_config.json")
vocab_path = os.path.join(os.path.dirname(__file__), "vocab.txt")

# Load ONNX model
ort_session = onnxruntime.InferenceSession(model_path)

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


