# Frugal

**A cost-aware search planner for SerpApi.**

Most agents search badly. They take a question, fire a single query at one
engine, take the top ten results, and stuff them into a context window. When
that fails to answer the question they fire another query, and another, until
something sticks or the budget is gone.

Frugal treats retrieval as a planning problem instead. Given a question and an
explicit budget, it compiles a *search plan*: which of SerpApi's engines to
call, in what order, with which query reformulations — then executes that plan,
stopping early once additional searches stop adding evidence.

The claim is that this answers questions at least as well as naive search while
issuing materially fewer billable searches. That claim is measured, not
asserted: see [Results](#results).

> **Status:** in development. The planner and benchmark are landing through
> early October 2026.

## Why this exists

SerpApi bills per search. Every wasted call is real money, and naive agents
waste a great many of them:

- **Redundant searches.** The same query, reformulated slightly, returns
  substantially the same page. Nobody notices because nobody is counting.
- **Wrong engine.** A question about a paper goes to web search when
  `google_scholar` would have answered it in one call.
- **No stopping rule.** An agent that has already found the answer keeps
  searching, because nothing tells it the evidence has saturated.

Each of those is a planning failure, and each is measurable.

## How it works

```
question
   │
   ├─ classify ──────▶ what kind of question is this?
   │
   ├─ route ─────────▶ which engines can answer it, at what cost?
   │
   ├─ reformulate ───▶ which query variants are worth issuing?
   │
   ├─ execute ───────▶ under a hard budget, concurrently
   │                     │
   │                     └─ saturation check: is new evidence still arriving?
   │                            └─ no ──▶ stop early, return the budget
   │
   └─ normalised, deduplicated, provenance-tagged evidence
```

Every stage is separately testable and separately ablatable, which is what makes
the benchmark able to attribute the improvement to a specific mechanism rather
than to the system as a whole.

## The cache is load-bearing

Frugal records every SerpApi response to disk, addressed by a hash of the engine
and its canonicalised parameters. This is not an optimisation:

- **Each unique search is paid for once, ever.** Parameter order, `None` values
  and credentials are normalised out of the key, so two spellings of the same
  search do not bill twice.
- **Benchmarks re-run at zero cost.** In replay mode the cache serves only from
  disk and raises on a miss rather than silently reaching for the network — so a
  recorded fixture set either covers a workload completely or fails loudly.
- **Results reproduce without an API key.** The fixtures behind the results
  table are committed. Clone the repository and re-run the benchmark; you should
  get the same numbers, having spent nothing.

Credentials never reach disk: they are stripped from both the key derivation and
the stored record.

## Results

Pending. This section will carry the head-to-head table — searches issued,
answer quality, and wall-clock latency, for naive search against planned search
across the benchmark question set — together with per-mechanism ablations.

It will not be filled in with anything that has not been measured.

## Install

Requires Python 3.11 or newer.

```bash
git clone https://github.com/desilva23/frugal.git
cd frugal
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Then add your SerpApi key. A free key gives 250 searches per month, which is
enough to try it:

```bash
cp .env.example .env
```

Open `.env` and replace the placeholder with your key from
[serpapi.com/manage-api-key](https://serpapi.com/manage-api-key):

```
SERPAPI_API_KEY=your-actual-key
```

`.env` is gitignored, so the key is never committed. Confirm the setup:

```bash
frugal doctor
```

## Reproducing the benchmark

The recorded fixtures are committed, so this needs no key and spends nothing:

```bash
pytest                          # test suite
python -m frugal.benchmark      # replay the benchmark from fixtures
```

## Which SerpApi engines are used

Frugal routes across engines rather than defaulting to web search: `google`,
`google_news`, `google_scholar`, `google_patents`, `google_shopping`,
`google_jobs`, `google_trends`, `google_maps`, `google_finance`. Each carries
cost, latency and volatility metadata that the planner reads when deciding
where a question should go and how long its answer stays fresh.

## Licence

MIT — see [LICENSE](LICENSE).
