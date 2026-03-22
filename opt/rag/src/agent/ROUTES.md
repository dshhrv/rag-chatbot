route_intent

retrieve_search

judge_search

retry_search

retrieve_comparison

judge_comparison

retry_comparison

generate_search_answer

generate_comparison_answer

draft_email

clarify

refuse

escalate


START
 -> route_intent

route_intent
 -> SEARCH -> retrieve_search
 -> COMPARISON -> retrieve_comparison
 -> EMAIL -> draft_email
 -> CLARIFY -> clarify
 -> REFUSE -> refuse

retrieve_search -> judge_search
judge_search
 -> ok -> generate_search_answer
 -> fail & retry_count == 0 -> retry_search
 -> fail & retry_count >= 1 -> escalate

retry_search -> judge_search

retrieve_comparison -> judge_comparison
judge_comparison
 -> ok -> generate_comparison_answer
 -> fail & retry_count == 0 -> retry_comparison
 -> fail & retry_count >= 1 -> escalate

retry_comparison -> judge_comparison

draft_email -> END
clarify -> END
refuse -> END
generate_search_answer -> END
generate_comparison_answer -> END
escalate -> END