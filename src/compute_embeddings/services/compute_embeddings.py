import os
import json
import numpy as np
import onnxruntime
from transformers import BertTokenizer, BertConfig
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Define paths relative to the current script file

model_path = os.path.join(os.path.dirname(__file__),'model_optimized.onnx')
    # Load ONNX model
with open(model_path, 'r') as f:
    ort_session = onnxruntime.InferenceSession(f.read())
#else:
#    import s3fs
#    region = 'us-east-1'
#    s3 = s3fs.S3FileSystem()
#    s3_file_path = 's3://weightsforemb/model_optimized.onnx'
#    with s3.open(s3_file_path, 'rb') as f:
#        ort_session = onnxruntime.InferenceSession(f.read())

config_path = os.path.join(os.path.dirname(__file__), "config.json")
ort_config_path = os.path.join(os.path.dirname(__file__), "ort_config.json")
special_tokens_path = os.path.join(os.path.dirname(__file__), "special_tokens_map.json")
tokenizer_path = os.path.join(os.path.dirname(__file__), "tokenizer.json")
tokenizer_config_path = os.path.join(os.path.dirname(__file__), "tokenizer_config.json")
vocab_path = os.path.join(os.path.dirname(__file__), "vocab.txt")

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
    cls_embedding = outputs[0][:, 0, :]  # Extract embeddings for the [CLS] token
    return cls_embedding.astype(np.float16).reshape((-1,))
