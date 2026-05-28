"""
Implementation of 3 methods for multilingual question output.

"""
from utils import generate_text
from repl_system import REPLEnvironment


class DirectMultilingualRAG:

    def __init__(self, model, tokenizer):

        self.model = model
        self.tokenizer = tokenizer

    def answer(
        self,
        question,
        en_context,
        zh_context,
        target_lang="Vietnamese"
    ):

        prompt = f"""
        Answer the question in {target_lang}.

        {en_context}

        {zh_context}

        Question:
        {question}

        Answer in {target_lang}.
        """

        return generate_text(
            self.model,
            self.tokenizer,
            prompt
        )

from deep_translator import GoogleTranslator
from utils import generate_text


class TranslateThenGenerate:

    """
    Translate-first multilingual QA.

    Pipeline:
    1. Translate ZH context → English
    2. Combine EN + translated EN
    3. Generate final answer directly in Vietnamese
    """

    def __init__(self, model, tokenizer):

        self.model = model
        self.tokenizer = tokenizer

    def translate_to_english(self, text):

        translated = GoogleTranslator(
            source="zh-CN",
            target="en"
        ).translate(text)

        return translated

    def answer(
        self,
        question,
        en_context,
        zh_context,
        target_lang="Vietnamese"
    ):

        # ============================================
        # STEP 1 — TRANSLATE CHINESE CONTEXT
        # ============================================

        zh_to_en = self.translate_to_english(
            zh_context
        )

        # ============================================
        # STEP 2 — BUILD UNIFIED ENGLISH CONTEXT
        # ============================================

        combined_context = f"""
English Context:
{en_context}

Translated Chinese Context:
{zh_to_en}
"""

        # ============================================
        # STEP 3 — FINAL QA
        # ============================================

        final_prompt = f"""
You are a multilingual question answering assistant.

Use the context below to answer the question.

Context:
{combined_context}

Question:
{question}

Rules:
- Answer ONLY in {target_lang}
- Be concise
- Use factual information only
- Avoid code-switching
"""

        final_answer = generate_text(
            self.model,
            self.tokenizer,
            final_prompt
        )

        return final_answer
    
class TranslateThenGenerateOld:

    def __init__(self, model, tokenizer):

        self.model = model
        self.tokenizer = tokenizer

    def translate(self, text, source_lang, target_lang):

        prompt = f"""
        Translate the following text from {source_lang}
        to {target_lang}.

        Text:
        {text}
        """

        return generate_text(
            self.model,
            self.tokenizer,
            prompt
        )

    def answer(
        self,
        question,
        en_context,
        zh_context,
        target_lang="Vietnamese"
    ):

        zh_to_en = self.translate(
            zh_context,
            "Chinese",
            "English"
        )

        combined_context = en_context + "\n" + zh_to_en

        answer_prompt = f"""
        Use the following context to answer the question.

        Context:
        {combined_context}

        Question:
        {question}

        Answer in English.
        """

        english_answer = generate_text(
            self.model,
            self.tokenizer,
            answer_prompt
        )

        final_answer = self.translate(
            english_answer,
            "English",
            target_lang
        )

        return final_answer

class RLMInspiredXLQA:

    """
    RLM-inspired hierarchical multilingual QA.

    Instead of generating executable Python,
    the root orchestrator directly delegates
    isolated tasks to sub-LLM instances.

    Key idea:
    - EN context processed independently
    - ZH context processed independently
    - Intermediate summaries stored
    - Final synthesis performed afterward

    This preserves:
    - context isolation
    - hierarchical reasoning
    - modular delegation

    while avoiding fragile code generation.
    """

    def __init__(self, model, tokenizer):

        self.model = model
        self.tokenizer = tokenizer

        self.repl = REPLEnvironment(
            model,
            tokenizer
        )

    def answer(
        self,
        question,
        en_context,
        zh_context,
        target_lang="Vietnamese"
    ):

        # ==================================================
        # STEP 1 — ENGLISH SUB-LLM
        # ==================================================

        en_summary = self.repl.sub_llm(
            context=en_context,
            instruction=f"""
Summarise the context and extract information
relevant to answering the question.

Question:
{question}

Respond ONLY in {target_lang}.
"""
        )

        # ==================================================
        # STEP 2 — CHINESE SUB-LLM
        # ==================================================

        zh_summary = self.repl.sub_llm(
            context=zh_context,
            instruction=f"""
概括上下文并提取与回答问题相关的信息.

问题:
{question}

仅回复 {target_lang}.
"""
        )

        # ==================================================
        # STEP 3 — STORE INTERMEDIATE MEMORY
        # ==================================================

        memory = f"""
English Summary:
{en_summary}

Chinese Summary:
{zh_summary}
"""

        # ==================================================
        # STEP 4 — FINAL SYNTHESIS
        # ==================================================

        final_prompt = f"""
Bạn là một hệ thống suy luận đa ngôn ngữ.

Bạn được cung cấp các tóm tắt riêng biệt được trích xuất

Sử dụng các tóm tắt để trả lời câu hỏi.

Câu hỏi:
{question}

Tóm tắt:
{memory}

Hướng dẫn:
- Trả lời CHỈ trong {target_lang}
- Kết hợp thông tin từ cả hai tóm tắt
- Tránh code-switching
- Đưa ra câu trả lời ngắn gọn, dựa trên sự thật
"""

        final_answer = generate_text(
            self.model,
            self.tokenizer,
            final_prompt
        )

        return final_answer
class RLMInspiredXLQAnew:

    """
    RLM-inspired isolated evidence extraction system.
    """

    def __init__(self, model, tokenizer):

        self.model = model
        self.tokenizer = tokenizer

        self.repl = REPLEnvironment(
            model,
            tokenizer
        )

    def answer(
        self,
        question,
        en_context,
        zh_context,
        target_lang="Vietnamese"
    ):

        # ============================================
        # STEP 1 — ENGLISH EVIDENCE EXTRACTION
        # ============================================

        en_evidence = self.repl.sub_llm(
            context=en_context,
            instruction=f"""
You are an evidence extraction system.

Task:
Extract ONLY the information useful for answering the question.

Question:
{question}

Rules:
- Do NOT summarize generally
- Preserve factual details
- Preserve names, dates, locations
- Use concise bullet points
- Respond ONLY in {target_lang}
"""
        )

        # ============================================
        # STEP 2 — CHINESE EVIDENCE EXTRACTION
        # ============================================

        zh_evidence = self.repl.sub_llm(
            context=zh_context,
            instruction=f"""
你是一个信息提取系统。

任务：
只提取与回答问题相关的信息。

问题：
{question}

规则：
- 不要做一般性总结
- 保留事实细节
- 保留名称、日期、地点
- 使用简洁要点
- 只使用 {target_lang} 回答
"""
        )

        # ============================================
        # STEP 3 — FINAL REASONING
        # ============================================

        final_prompt = f"""
You are a multilingual reasoning assistant.

Question:
{question}

Evidence from English context:
{en_evidence}

Evidence from Chinese context:
{zh_evidence}

Task:
Use the evidence above to answer the question.

Rules:
- Respond ONLY in {target_lang}
- Do NOT mention the evidence sources
- Avoid repetition
- Give a direct factual answer
"""

        final_answer = generate_text(
            self.model,
            self.tokenizer,
            final_prompt
        )

        return final_answer