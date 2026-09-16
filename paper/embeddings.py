"""MiniLM ONNX inference: attention-masked mean pooling and L2 normalization."""
import threading
import numpy as np

class OffsetTokenizer:
    def __init__(self, path):
        from tokenizers import Tokenizer
        self.raw = Tokenizer.from_file(path)
        self.raw.no_truncation()
        self.raw.no_padding()

    def num_special_tokens_to_add(self, pair=False):
        return self.raw.num_special_tokens_to_add(pair)

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=True, truncation=False):
        return {'offset_mapping': self.raw.encode(text, add_special_tokens=add_special_tokens).offsets}

class MiniLMOnnx:
    max_seq_length = 256

    def __init__(self, model_name):
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer
        import onnxruntime as ort
        tokenizer_path = hf_hub_download(model_name, 'tokenizer.json', token=False)
        model_path = hf_hub_download(model_name, 'onnx/model.onnx', token=False)
        self.tokenizer = OffsetTokenizer(tokenizer_path)
        self.batch_tokenizer = Tokenizer.from_file(tokenizer_path)
        self.batch_tokenizer.enable_truncation(max_length=self.max_seq_length)
        self.batch_tokenizer.enable_padding(pad_id=0, pad_token='[PAD]')
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        self.session = ort.InferenceSession(model_path, sess_options=options, providers=['CPUExecutionProvider'])
        self.lock = threading.Lock()

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        outputs = []
        with self.lock:
            for start in range(0, len(texts), 16):
                batch = self.batch_tokenizer.encode_batch(texts[start:start+16])
                values = {
                    'input_ids': np.asarray([x.ids for x in batch], dtype=np.int64),
                    'attention_mask': np.asarray([x.attention_mask for x in batch], dtype=np.int64),
                    'token_type_ids': np.asarray([x.type_ids for x in batch], dtype=np.int64),
                }
                feeds = {x.name: values[x.name] for x in self.session.get_inputs()}
                tokens = self.session.run(None, feeds)[0]
                mask = values['attention_mask'][..., None]
                pooled = (tokens * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1)
                if normalize_embeddings:
                    pooled /= np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)
                outputs.append(pooled.astype(np.float32))
        return np.concatenate(outputs) if outputs else np.empty((0, 384), dtype=np.float32)
