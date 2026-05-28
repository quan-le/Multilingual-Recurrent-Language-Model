import io
import contextlib

class REPLEnvironment:

    def __init__(self, model, tokenizer):

        self.model = model
        self.tokenizer = tokenizer

    def sub_llm(self, context, instruction):

        from utils import generate_text

        prompt = f"""
        Context:
        {context}

        Instruction:
        {instruction}
        """

        return generate_text(
            self.model,
            self.tokenizer,
            prompt
        )

    def execute_script(self, code, variables):

        local_env = {}

        local_env.update(variables)

        local_env["sub_llm"] = self.sub_llm

        stdout_capture = io.StringIO()

        try:
            with contextlib.redirect_stdout(stdout_capture):
                exec(code, {}, local_env)

            result = local_env.get("Final", "")

            return result

        except Exception as e:
            return f"EXECUTION_ERROR: {str(e)}"