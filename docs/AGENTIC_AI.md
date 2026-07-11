# Agentic AI support

The core agent is provider-neutral and can operate offline through deterministic task parsing and rule-based recommendation. An external LLM can be injected as a callable backend without coupling the library to one vendor.

## Agent tools
1. list and inspect datasets;
2. discover Hugging Face resources;
3. inspect model capabilities and limitations;
4. validate task/model compatibility;
5. recommend models with reasons and cautions;
6. build benchmark splits, metrics, and ablations;
7. emit a pinned run manifest.

```python
from navier_cfd.agents import AgentOrchestrator
agent = AgentOrchestrator()
plan = agent.plan("Benchmark sim-to-real cylinder wake forecasting on RealPDEBench, 24 GB GPU, conservation and uncertainty required")
print(plan.to_dict())
```

The library never sends files, credentials, or datasets to an LLM by default.
