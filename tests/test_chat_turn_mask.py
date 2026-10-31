from ghostlm.tokenizer import GhostTokenizer


def test_untrained_assistant_turn_is_context_only():
    tok = GhostTokenizer()
    turns = [
        {"role": "user", "content": "what is 2+2"},
        {"role": "assistant", "content": "5", "train": False},
        {"role": "user", "content": "no, 4"},
        {"role": "assistant", "content": "You are right, 4."},
    ]
    ids, mask = tok.encode_chat(turns)
    plain, plain_mask = tok.encode_chat([{k: v for k, v in t.items() if k != "train"} for t in turns])
    assert ids == plain
    assert sum(mask) < sum(plain_mask)
    end = tok._special_tokens[tok.END]
    ends = [i for i, t in enumerate(ids) if t == end]
    first_assistant_end, last_end = ends[1], ends[3]
    assert mask[first_assistant_end] == 0 and all(m == 0 for m in mask[: first_assistant_end + 1])
    assert mask[last_end] == 1
