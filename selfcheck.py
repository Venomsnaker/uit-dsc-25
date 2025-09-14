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
        self.output_dict = {
            'yes': 1,
            'n/a': 0.5,
            'no': 0
        }
        self.not_defined = 'n/a'
        
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
    
    def _postprocess(self, response):
        response = response.lower().strip()

        for output in self.output_dicts:
            if output in response:
                response = output
        if response not in self.output_dicts.keys():
            response = 'n/a'
        return self.output_dict[response]

    def predict_intrinsic(self, context: str, sentence: str):
        prompt = self.intrinsic_prompt_template.format(context=context, sentence=sentence)
        return self._postprocess(self._generate(prompt, max_new_tokens=4))
    
    def predict_extrinsic(self, context: str, sentence: str):
        prompt = self.extrinsic_prompt_template.format(context=context, sentence=sentence)
        return self._postprocess(self._generate(prompt, max_new_tokens=4))


    
