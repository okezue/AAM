# Closed-model experimental path

## Feasible claim

A closed reader can consume source text selected by an AAM written in an open surrogate feature
space. This tests whether recurrent associative retrieval identifies better evidence than dense RAG
for the same downstream reader.

Configured reader adapters:

- OpenAI Responses API;
- Anthropic Messages API;
- Google Gemini GenerateContent API.

Credentials are read from environment variables. They are never written to configs or artifacts.
Generation records include the provider model string, available response/request ID, usage, stop or
finish metadata, and elapsed time.

## Infeasible through ordinary public APIs

The text APIs do not provide a supported general route to:

- capture arbitrary hidden activations at the required model site;
- apply a public SAE to the exact proprietary checkpoint;
- persist/reinsert internal K/V states;
- add internal memory tokens or cross-attention modules;
- update proprietary weights or fast-weight state with Hebbian rules.

A public autoencoder by itself would still be insufficient without the exact model revision,
tokenizer, activation site, normalization convention, and runtime activation interface.

Therefore L5 must be labelled **open-memory/closed-reader hybrid associative retrieval**, not
closed-model activation memory.

## Protocol

1. Freeze one open address encoder, including revision and feature cache.
2. Write memories before observing the query in query-independent conditions.
3. Cache retrieval outputs so all reader conditions receive the same episode candidates where
   appropriate.
4. Compare no memory, dense RAG, activation kNN, full AAM, and oracle evidence.
5. Fix retrieved top-k and maximum injected characters/tokens.
6. Use identical source labels, separators, and prompt instructions across retrieval methods.
7. Set temperature to zero when supported and fix output-token limit.
8. Record exact model string, request date/time, response ID, usage, stop reason, and refusal/error.
9. Cache requests by a complete cryptographic request hash where provider terms permit. The current
   reader wrappers record provenance but a durable response cache/retry scheduler is intentionally a
   host-side extension.
10. Blind automatic judges to condition names and preserve raw reader output.
11. Re-run a fixed stability subset on multiple dates because provider aliases may drift.

## Prompt boundary

Retrieved memory is quoted as untrusted source material with immutable source IDs. The system prompt
must say that instructions inside memories are data, not commands, and that factual claims require
source support. Include stored prompt-injection attacks in the evaluation.

## Cost control

Run in stages:

1. retrieval-only source metrics on all examples;
2. generation on a small pre-registered subset after retrieval passes;
3. expand to the full confirmatory set;
4. run oracle and no-memory references once per fixed reader version.

Log input/output tokens and provider-reported costs if available, but do not hardcode price tables in
research results because pricing changes. Cost is a secondary operational metric; evidence quality,
request count, and token use are stable quantities.
