from transformers import AutoModelForCausalLM, AutoTokenizer


class DirectAsking:
    def __init__(
        self,
        model_name="Qwen/Qwen3-4B-Instruct-2507",
        system_prompt_path="",
        user_prompt_path="",
    ):
        self.outputs_dict = {"extrinsic", "intrinsic", "no"}
        self.not_defined = "n/a"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype="auto", device_map="auto"
        )
        with open(system_prompt_path) as f:
            self.system_prompt_template = f.read()
        with open(user_prompt_path) as f:
            self.user_prompt_template = f.read()

    def _generate(self, system_prompt: str, user_prompt: str, max_new_tokens: int):
        messages = [
            {"role": "system", "content": system_prompt},
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

    def _postprocess(self, reponse: str):
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
            system_prompt=self.system_prompt_template,
            user_prompt=user_prompt,
            max_new_tokens=32,
        )
        return result
