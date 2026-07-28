from opentelemetry import trace

tracer = trace.get_tracer("llm-zoomcamp")

INSTRUCTIONS = """
Your task is to answer questions about the course based only on the
provided context.

If the answer cannot be found in the context, respond with
"I don't know."
"""

PROMPT_TEMPLATE = """
QUESTION:
{question}

CONTEXT:
{context}
""".strip()


class RAGBase:

    def __init__(
        self,
        index,
        llm_client,
        instructions=INSTRUCTIONS,
        prompt_template=PROMPT_TEMPLATE,
        model="gpt-5.4-mini",
    ):
        self.index = index
        self.llm_client = llm_client
        self.instructions = instructions
        self.prompt_template = prompt_template
        self.model = model

    def search(self, query, num_results=5):
        return self.index.search(
            query,
            num_results=num_results,
        )

    def build_context(self, search_results):
        lines = []

        for doc in search_results:
            lines.append(f"FILE: {doc['filename']}")
            lines.append(doc["content"])
            lines.append("")

        return "\n".join(lines).strip()

    def build_prompt(self, query, search_results):
        context = self.build_context(search_results)

        return self.prompt_template.format(
            question=query,
            context=context,
        )

    def llm(self, prompt):
        input_messages = [
            {
                "role": "developer",
                "content": self.instructions,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        response = self.llm_client.responses.create(
            model=self.model,
            input=input_messages,
        )

        return response.output_text

    def rag(self, query):
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        return self.llm(prompt)


class GeminiRAG(RAGBase):

    def __init__(
        self,
        index,
        llm_client,
        instructions=INSTRUCTIONS,
        prompt_template=PROMPT_TEMPLATE,
        model="gemini-2.5-flash",
    ):
        super().__init__(
            index=index,
            llm_client=llm_client,
            instructions=instructions,
            prompt_template=prompt_template,
            model=model,
        )

    def llm(self, prompt):
        response = self.llm_client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={
                "system_instruction": self.instructions,
            },
        )

        return response.text


class RAGTraced:

    def search(self, query, num_results=5):
        with tracer.start_as_current_span("search") as span:
            span.set_attribute("query", query)
            span.set_attribute("num_results", num_results)

            results = super().search(query, num_results)

            span.set_attribute("results_count", len(results))
            return results

    def llm(self, prompt):
        with tracer.start_as_current_span("llm") as span:
            span.set_attribute("model", self.model)
            span.set_attribute("prompt_length", len(prompt))

            answer = super().llm(prompt)

            span.set_attribute("response_length", len(answer))
            return answer

    def rag(self, query):
        with tracer.start_as_current_span("rag") as span:
            span.set_attribute("query", query)
            span.set_attribute("model", self.model)

            answer = super().rag(query)

            span.set_attribute("answer_length", len(answer))
            return answer


class GeminiRAGTraced(RAGTraced, GeminiRAG):

    def llm(self, prompt):
        with tracer.start_as_current_span("llm") as span:
            span.set_attribute("model", self.model)
            span.set_attribute("prompt_length", len(prompt))

            response = self.llm_client.models.generate_content(
                model=self.model,
                contents=prompt,
                config={
                    "system_instruction": self.instructions,
                },
            )

            usage = response.usage_metadata

            # Save usage so monitoring.py can access it later.
            self.last_usage = usage

            span.set_attribute(
                "input_tokens",
                usage.prompt_token_count,
            )
            span.set_attribute(
                "output_tokens",
                usage.candidates_token_count,
            )
            span.set_attribute(
                "total_tokens",
                usage.total_token_count,
            )

            answer = response.text or ""

            span.set_attribute(
                "response_length",
                len(answer),
            )

            return answer
                
class OpenAIRAGTraced(RAGTraced, RAGBase):
    pass


# from opentelemetry import trace

# tracer = trace.get_tracer("llm-zoomcamp")

# INSTRUCTIONS = '''
# Your task is to answer questions from the course participants
# based on the provided context.

# Use the context to find relevant information and provide accurate
# answers. If the answer is not found in the context,
# respond with "I don't know."
# '''

# PROMPT_TEMPLATE = '''
# QUESTION: {question}

# CONTEXT:
# {context}
# '''.strip()


# class RAGBase:

#     def __init__(
#         self,
#         index,
#         llm_client,
#         instructions=INSTRUCTIONS,
#         prompt_template=PROMPT_TEMPLATE,
#         course='llm-zoomcamp',
#         model='gpt-5.4-mini'
#     ):
#         self.index = index
#         self.llm_client = llm_client
#         self.instructions = instructions
#         self.course = course
#         self.prompt_template = prompt_template
#         self.model = model

#     def search(self, query, num_results=5):
#         boost_dict = {'question': 3.0, 'section': 0.5}
#         # filter_dict = {'course': self.course}

#         return self.index.search(
#             query,
#             num_results=num_results,
#             boost_dict=boost_dict,
#             # filter_dict=filter_dict
#         )

#     def build_context(self, search_results):
#         lines = []

#         for doc in search_results:
#             lines.append(doc['section'])
#             lines.append('Q: ' + doc['question'])
#             lines.append('A: ' + doc['answer'])
#             lines.append('')

#         return '\n'.join(lines).strip()

#     def build_prompt(self, query, search_results):
#         context = self.build_context(search_results)
#         return self.prompt_template.format(
#             question=query, context=context
#         )

#     def llm(self, prompt):
#         input_messages = [
#             {'role': 'developer', 'content': self.instructions},
#             {'role': 'user', 'content': prompt}
#         ]

#         response = self.llm_client.responses.create(
#             model=self.model,
#             input=input_messages
#         )

#         return response.output_text

#     def rag(self, query):
#         search_results = self.search(query)
#         prompt = self.build_prompt(query, search_results)
#         answer = self.llm(prompt)
#         return answer


# class RAGTraced:
#     def search(self, query, num_results=5):
#         with tracer.start_as_current_span("search") as span:
#             span.set_attribute("query", query)
#             span.set_attribute("num_results", num_results)

#             results = super().search(query, num_results)

#             span.set_attribute("results_count", len(results))
#             return results

#     def llm(self, prompt):
#         with tracer.start_as_current_span("llm") as span:
#             span.set_attribute("model", self.model)
#             span.set_attribute("prompt_length", len(prompt))

#             answer = super().llm(prompt)

#             span.set_attribute("response_length", len(answer))
#             return answer

#     def rag(self, query):
#         with tracer.start_as_current_span("rag") as span:
#             span.set_attribute("query", query)
#             span.set_attribute("model", self.model)

#             answer = super().rag(query)

#             span.set_attribute("answer_length", len(answer))
#             return answer

# class GeminiRAG(RAGBase):

#     def __init__(
#         self,
#         index,
#         llm_client,
#         instructions=INSTRUCTIONS,
#         prompt_template=PROMPT_TEMPLATE,
#         course='llm-zoomcamp',
#         model='gemini-2.5-flash'
#     ):
#         super().__init__(
#             index=index,
#             llm_client=llm_client,
#             instructions=instructions,
#             prompt_template=prompt_template,
#             course=course,
#             model=model
#         )

#     def llm(self, prompt):

#         response = self.llm_client.models.generate_content(
#                 model="gemini-2.5-flash",
#                 contents=prompt,
#                 config={
#                 "system_instruction": self.instructions
#                 }
#             )
        
#         return response.text
    
# class GeminiRAGTraced(RAGTraced, GeminiRAG):
#     pass



