import time

from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI

class DirectAsking:
    def __init__(
        self,
        model_name="Qwen/Qwen3-4B-Instruct-2507",
        user_prompt_path="",
    ):
        self.outputs_dict = {"extrinsic", "intrinsic", "no"}
        self.not_defined = "no"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype="auto", device_map="auto"
        )
        with open(user_prompt_path) as f:
            self.user_prompt_template = f.read()

    def _generate(self, user_prompt: str, max_new_tokens: int):
        messages = [
            {"role": "user", "content": user_prompt},
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self.tokenizer([inputs], return_tensors="pt").to(
            self.model.device
        )
        generated_ids = self.model.generate(
            **model_inputs, max_new_tokens=max_new_tokens
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
        content = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        return content

    def _postprocess(self, response: str):
        response = response.lower().strip()

        for output in self.outputs_dict:
            if output in response:
                return output
        return self.not_defined

    def predict(self, context: str, prompt: str, response: str):
        user_prompt = self.user_prompt_template.format(
            context=context, prompt=prompt, response=response
        )
        result = self._generate(
            user_prompt=user_prompt,
            max_new_tokens=16,
        )
        return result

class DirectAskingAPI:
    def __init__(
        self,
        model='gpt-4o-mini-2024-07-18',
        api_key='',
        user_prompt_path='',
        retries=3,
    ):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        with open(user_prompt_path) as f:
            self.user_prompt_template = f.read()
        self.retries = retries
        self.outputs_dict = {"extrinsic", "intrinsic", "no"}
        self.not_defined = "no"
    
    def _generate(self, user_prompt: str, max_tokens=16):
        for attempt in range(self.retries):
            try:
                request_kwargs = {
                    'model': self.model,
                    'messages': [{'role': 'user', 'content': user_prompt}],
                    'max_completion_tokens': max_tokens,
                }

                response = self.client.chat.completions.create(**request_kwargs)
                return response.choices[0].message.content
            except Exception as e:
                if attempt <= self.retries:
                    time.sleep(1)
                else:
                    raise e
                
    def _postprocess(self, response: str):
        response = response.lower().strip()

        for output in self.outputs_dict:
            if output in response:
                return output
        return self.not_defined
                
    def predict(self, context: str, prompt: str, response: str):
        user_prompt = self.user_prompt_template.format(
            context=context, prompt=prompt, response=response
        )
        result = self._generate(
            user_prompt=user_prompt,
            max_tokens=16,
        )
        return self._postprocess(result) 