from agent_router import run_agent_timed

queries = [
    "Что такое Асинхронное освоение Дисциплины?"
]




for q in queries:
    state, latency = run_agent_timed(q, "ru")
    answer = state.get('answer')
    print(f"Query: {q}")
    print(f"Value: {answer}")
    if isinstance(answer, dict):
        print(f"  found: {answer.get('found')}")
        print(f"  citations: {answer.get('citations')}")
    print(f"  Latency: {latency:.3f}s\n")