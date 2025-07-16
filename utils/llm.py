from transformers import Gemma3nForConditionalGeneration, AutoTokenizer
import torch
import gc
import os

class LLM:
    def __init__(self, model_name_or_path="google/gemma-3n-e4b-it", device=None, pipe=None, tokenizer=None):
        # Kiểm tra RAM khả dụng
        print("load_check_point", model_name_or_path)
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if pipe is not None and tokenizer is not None:
            self.model = pipe
            self.tokenizer = tokenizer
        else:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            if torch.cuda.is_available():
                if torch.cuda.is_bf16_supported():
                    dtype = torch.bfloat16
                else:
                    dtype = torch.float16

                print("Using CUDA.")

                self.model = Gemma3nForConditionalGeneration.from_pretrained(
                    model_name_or_path,
                    device_map="auto",
                    torch_dtype=dtype,
                ).eval()
            else:
                print("Using CPU.")
                self.model = Gemma3nForConditionalGeneration.from_pretrained(
                    model_name_or_path,
                    device_map="cpu",
                    torch_dtype=torch.float32,
                ).eval()

        # self.model = self.model.to(self.device)
        # self.model.eval()

    def generate(self, prompt, max_new_tokens=100, temperature=0.7):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        result = self.tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"result: {result}")
        return result[len(prompt):].strip() if result.startswith(prompt) else result.strip()
