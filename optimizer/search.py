from optimizer.constraints import VLLMCandidate

def generate_candidates(baseline):
    base = VLLMCandidate.model_validate(baseline)
    candidates = [
        base.model_copy(update={"max_num_seqs": min(512, base.max_num_seqs * 2)}),
        base.model_copy(update={"max_num_batched_tokens": min(65536, base.max_num_batched_tokens * 2)}),
        base.model_copy(update={"enable_chunked_prefill": not base.enable_chunked_prefill}),
    ]
    result, seen = [], set()
    for item in candidates:
        value = item.model_dump()
        key = tuple(sorted(value.items()))
        if value != base.model_dump() and key not in seen:
            result.append(value)
            seen.add(key)
    return result
