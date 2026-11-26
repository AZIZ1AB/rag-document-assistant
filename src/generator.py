import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LocalAnswerGenerator:
    """Lazy-loading local Hugging Face instruction-model generator."""

    def __init__(self, model_name):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None

    def load(self):
        if self.model is not None:
            return

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # CPU is the safest default for broad compatibility. Users with a
        # supported GPU can configure device_map through their own setup.
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
        )
        self.model.eval()

    def answer(self, question, retrieved_chunks, max_new_tokens=220):
        if not retrieved_chunks:
            return "I could not find relevant text in the indexed documents."

        self.load()

        context_parts = []
        for number, item in enumerate(retrieved_chunks, start=1):
            context_parts.append(
                f"[{number}] Source: {item['source']}, page {item['page']}\n"
                f"{item['text']}"
            )
        context = "\n\n".join(context_parts)

        messages = [
            {
                "role": "system",
                "content": (
                    "Answer using only the supplied document context. "
                    "If the context does not contain the answer, say there is "
                    "not enough information in the documents. Do not invent "
                    "facts. Refer to source numbers when possible."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
