from transformers import AutoModelForCausalLM, AutoTokenizer

class SelfCheckPrompt:
    def __init__(
        self,
        model_name = 'Qwen/Qwen3-4B-Instruct-2507',
        intrinsic_prompt_path = '',
        extrinsic_prompt_path = '',
    ):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype='auto', device_map='auto'
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        with open(intrinsic_prompt_path) as f:
            self.intrinsic_prompt_template = f.read()
        with open(extrinsic_prompt_path) as f:
            self.extrinsic_prompt_template = f.read()
        
    def _generate(self, prompt: str, max_new_tokens:int):
        messages = [
            {'role': 'user', 'content': prompt},
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self.tokenizer([inputs], return_tensors='pt').to(self.model.device)
        generated_ids = self.model.generate(**model_inputs, max_new_tokens=max_new_tokens)
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
        content = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        return content

    
