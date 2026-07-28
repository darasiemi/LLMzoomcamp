from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)

from starter import rag
from exporter import SQLiteSpanExporter


def calc_price(usage):
    """Calculate Gemini 2.5 Flash cost in USD."""

    input_price_per_million = 0.30
    output_price_per_million = 2.50

    input_tokens = usage.prompt_token_count or 0
    output_tokens = usage.candidates_token_count or 0

    input_cost = (
        input_tokens / 1_000_000
    ) * input_price_per_million

    output_cost = (
        output_tokens / 1_000_000
    ) * output_price_per_million

    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": input_cost + output_cost,
    }


def calc_total_price(usages):
    return sum(
        calc_price(usage)["total_cost"]
        for usage in usages
    )


provider = TracerProvider()

provider.add_span_processor(
    SimpleSpanProcessor(SQLiteSpanExporter("traces.db"))
)

# provider.add_span_processor(
#     SimpleSpanProcessor(
#         ConsoleSpanExporter()
#     )
# )

trace.set_tracer_provider(provider)

tracer = trace.get_tracer("llm-zoomcamp")

query = (
    "How does the agentic loop keep calling the model until it stops?"
)

with tracer.start_as_current_span("homework5") as span:
    span.set_attribute("query", query)

    answer = rag.rag(query)

    span.set_attribute("answer", answer)

    usage = rag.last_usage

    if usage is not None:
        input_tokens = usage.prompt_token_count or 0
        output_tokens = usage.candidates_token_count or 0
        total_tokens = usage.total_token_count or 0

        price = calc_price(usage)

        span.set_attribute(
            "input_tokens",
            input_tokens,
        )
        span.set_attribute(
            "output_tokens",
            output_tokens,
        )
        span.set_attribute(
            "total_tokens",
            total_tokens,
        )

        span.set_attribute(
            "input_cost",
            price["input_cost"],
        )
        span.set_attribute(
            "output_cost",
            price["output_cost"],
        )
        span.set_attribute(
            "total_cost",
            price["total_cost"],
        )

        print(f"Input tokens: {input_tokens}")
        print(f"Output tokens: {output_tokens}")
        print(f"Total tokens: {total_tokens}")
        print(f"Input cost: ${price['input_cost']:.8f}")
        print(f"Output cost: ${price['output_cost']:.8f}")
        print(f"Total cost: ${price['total_cost']:.8f}")
    else:
        print("Gemini usage metadata was not returned.")

print(answer)