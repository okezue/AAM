# External baselines

No third-party code is vendored. `baselines.yaml` records upstream repositories needed for
comparisons. Before a real run, execute:

```bash
python scripts/resolveexternalrevisions.py
```

Commit the generated `REVISION_LOCK.json` with the experiment artifacts, then clone exactly those
commits into isolated environments. License terms remain those of each upstream project.

## JSON subprocess adapter

Model-specific baselines can be connected without importing their dependency stack into AAM by using
`aamemory.integrations.JSONSubprocessBaseline`. The adapter:

1. writes one JSON request to a temporary input path;
2. substitutes `{input}` and `{output}` into an argument-vector command;
3. executes without a shell and with a configurable timeout;
4. requires the child process to write a JSON object to the output path;
5. records duration, stdout, and stderr for provenance.

This is the intended boundary for a pinned Activation Beacon checkout and other baselines whose
runtime dependencies conflict with the core repository. A baseline wrapper should report its exact
upstream commit, model revision, command, logical memory bytes, retrieved items, and timing fields.
It must fail rather than silently replacing an unavailable method with ordinary text RAG.
