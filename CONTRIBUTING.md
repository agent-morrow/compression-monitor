# Contributing to compression-monitor

Thanks for looking. This is an active research scaffold — contributions that sharpen the instrumentation, expand coverage, or stress-test the epistemological claims are welcome.

---

## Starter tasks (good first issues)

These are concrete, bounded, and don't require deep familiarity with the full codebase:

1. **Add a `--output-json` flag to `ghost_lexicon.py`**  
   Right now the script prints results to stdout. A structured JSON output makes it easier to pipe into dashboards or combine with other instrument outputs. See the existing output format and match the structure `behavioral_footprint.py` uses.

2. **Write a real test for the perturbation protocol**  
   `tests/` is sparse. Add a test that seeds a novel vocabulary term into a mock session, runs `ghost_lexicon.py` against before/after snapshots, and verifies the term appears in the decay report. Uses only stdlib + the existing script interface.

3. **Add a `--session-dir` flag to `behavioral_footprint.py`**  
   Currently takes a single log file. Accept a directory of session logs and compute drift across adjacent pairs. Useful for monitoring long-running agents.

4. **Document one real false negative**  
   Run `ghost_lexicon.py` on a session where you know compression happened (e.g., a Claude Code session with a compaction event). If the instrument reports clean and you can identify *why* the drift was missed, open an issue or PR documenting the failure mode. Negative results are useful.

5. **Add a `run_isolation_experiment.py` scaffold**  
   See [Issue #4](https://github.com/agent-morrow/compression-monitor/issues/4) for the 2×2 isolation design. A minimal scaffold that can run Cell A (no compressor) and Cell B (compressor on) for a given agent harness would be a meaningful contribution.

---

## What belongs here vs. a separate project

- **Instrument improvements** (new signals, better precision, coverage of reasoning traces): fits here
- **Framework integrations** (LangGraph, AutoGen, new agent harnesses): fits here
- **Entirely separate measurement approach** (e.g., behavioral probing via canonical prompts): probably a separate tool, link from the Related Tools section
- **Framing-level compression mitigation**: fits in Issues as a research question; does not have an obvious implementation path yet (see [Issue #5](https://github.com/agent-morrow/compression-monitor/issues/5))

---

## Process

- Open an issue before large changes so we can align on scope.
- For small fixes and starter tasks, a PR is fine without prior discussion.
- The pre-registration protocol (`preregister.py`) is load-bearing — changes to it should come with rationale and updated documentation.
- Instrument outputs and the lead-lag protocol are the stable interface. Internal implementation is fair game.

---

## Running locally

```bash
git clone https://github.com/agent-morrow/compression-monitor
cd compression-monitor
pip install -e ".[dev]"

# Run a quick test
python quickstart.py
python examples/sdk_compaction_hook_demo.py --polling

# Run the test suite
pytest tests/

# Verify the package builds
python -m build
```

---

*Questions? Open an issue or reach out at [morrow.run](https://morrow.run).*
