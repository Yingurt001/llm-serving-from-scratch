# LLM Serving from Scratch

Building a small LLM inference engine by hand, 20 minutes a day, to understand what serving systems like vLLM actually do.

Each session ends with one commit: a piece of code, a number worked out by hand, or a short log entry. Each week ends with one write-up.

## Round 1 roadmap (Nov 2026 – Feb 2027)

| Weeks | Topic | Output |
|---|---|---|
| 1–2 | Transformer forward pass; prefill vs decode | KV cache sizes worked out by hand |
| 3–5 | `miniserve` v0: GPT-2 small forward pass in plain PyTorch, greedy decoding loop | Logits match the Hugging Face reference |
| 6–8 | KV cache | Decode reuses cached K/V; per-token cost before vs after |
| 9–11 | Batching: static, then continuous (iteration-level scheduling) | A scheduler that admits new requests mid-generation |
| 12–14 | Paged KV cache (block table + allocator) | Memory waste before vs after |
| 15–16 | Benchmark against vLLM on a GPU | Throughput / latency table and a write-up |

Round 2 (later): Triton / CUDA kernels, SGLang, an open-source contribution, tool calling for a local agent.

## Layout

- `NEXT.md`: the one thing to do next session
- `LOG.md`: one entry per session, newest first
- `days/`: worked calculations and small exercises
- `notes/`: weekly write-ups
- `miniserve/`: the engine (from week 3)

## References

- Vaswani et al. *Attention Is All You Need.* NeurIPS 2017.
- Radford et al. *Language Models are Unsupervised Multitask Learners.* OpenAI 2019. (GPT-2)
- Yu et al. *Orca: A Distributed Serving System for Transformer-Based Generative Models.* OSDI 2022. (continuous batching)
- Kwon et al. *Efficient Memory Management for Large Language Model Serving with PagedAttention.* SOSP 2023. (vLLM)
