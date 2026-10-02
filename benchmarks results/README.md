# RAG benchmark suite

This is the shared, architecture-neutral evaluator for Systems 1–4. It never supplies gold evidence or reference answers to a system. Each system is invoked through a command configured in `../benchmark/configs/systemN.yaml`; the command receives `{"query", "task", "config"}` on stdin and must print exactly one JSON result containing `answer`, `evidence`, `trace`, `usage`, and `system_metadata`.

Build the initial versioned dataset from the existing fixture:

```bash
python -m benchmark.datasets build --fixture "RAG system 2/benchmarks/fixture_questions.json"
python -m benchmark.datasets validate
```

Run paired evaluation after configuring adapters:

```bash
python -m benchmark.runner --systems system1 system2 system3 system4 --split test --limit 100 --seed 42
python -m benchmark.report --run-id <run_id>
```

Generated runs and reports are intentionally ignored. Do not tune against `splits/test.jsonl`.
