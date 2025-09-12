#### Direct Asking
- Add the context, prompt, and response to a template and directly get the label from the LLM.
- Result: Poor
    - `gpt4o_mini`: 1000 train samples
        - Precision: 0.70
        - Recall: 0.62
        - F1 Score: 0.55
    - `qwen3_4b`:
        - Precision: 0.63
        - Recall: 0.61
        - F1: 0.61

#### SelfCheckGPT (Adopted)