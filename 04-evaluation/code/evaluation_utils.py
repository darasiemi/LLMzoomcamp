import time

from tqdm.auto import tqdm
from google.genai import types
from rag_helper import GeminiRAG

def calc_price(usage):
    # Gemini 2.5 Flash (Standard API)
    INPUT_PRICE_PER_MILLION = 0.30
    OUTPUT_PRICE_PER_MILLION = 2.50

    input_tokens = usage.prompt_token_count
    output_tokens = usage.candidates_token_count

    input_cost = (input_tokens / 1_000_000) * INPUT_PRICE_PER_MILLION
    output_cost = (output_tokens / 1_000_000) * OUTPUT_PRICE_PER_MILLION

    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": input_cost + output_cost,
    }


def calc_total_price(usages):
    return sum(calc_price(u)["total_cost"] for u in usages)


def llm_structured(
    client,
    instructions,
    user_prompt,
    output_type,
    model="gemini-2.5-flash",
):

    response = client.models.generate_content(
        model=model,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=instructions,
            response_mime_type="application/json",
            response_schema=output_type,
        ),
    )

    return response.parsed, response.usage_metadata


def llm_structured_retry(
    client,
    instructions,
    user_prompt,
    output_type,
    model="gemini-2.5-flash",
    max_retries=3,
):

    for attempt in range(max_retries):
        try:
            return llm_structured(
                client,
                instructions,
                user_prompt,
                output_type,
                model=model,
            )
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)


class RAGWithUsage(GeminiRAG):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.usages = []
        self.last_usage = None

    def reset_usage(self):
        self.usages = []
        self.last_usage = None

    def search(self, query, num_results=5):
        boost_dict = {
            "question": 1.0,
            "answer": 2.0,
            "section": 0.1,
        }

        filter_dict = {
            "course": self.course,
        }

        return self.index.search(
            query,
            num_results=num_results,
            boost_dict=boost_dict,
            filter_dict=filter_dict,
        )

    def llm(self, prompt):

        response = self.llm_client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=self.instructions,
            ),
        )

        self.last_usage = response.usage_metadata
        self.usages.append(response.usage_metadata)

        return response.text

    def total_cost(self):
        return calc_total_price(self.usages)


def map_progress(pool, seq, f):
    results = []

    with tqdm(total=len(seq)) as progress:
        futures = []

        for el in seq:
            future = pool.submit(f, el)
            future.add_done_callback(lambda _: progress.update())
            futures.append(future)

        for future in futures:
            results.append(future.result())

    return results