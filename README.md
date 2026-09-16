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

The claim is that this reaches answers a single web search cannot, without
spending much more to do it. That claim is measured rather than asserted, on a
benchmark anyone can re-run from committed fixtures without an API key: see
[Results](#results).

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

Twelve questions with verifiable answers, scored on whether the retrieved
evidence actually contains the answer — not on how many results came back.
Scoring is exact string matching against marker terms, so there is no judge
model and nothing to take on trust.

| strategy | answered | recall | searches | searches/question |
|---|---|---|---|---|
| naive (one web search) | 10/12 | 83% | 12 | 1.0 |
| **planned** | **12/12** | **100%** | **20** | **1.7** |

The two the baseline misses are the two it cannot reach: a question about
whether interest is rising over time, which needs demand data rather than links,
and a question about who is hiring, which needs a jobs index. No number of web
searches answers either.

### Is the routing doing the work?

The obvious objection to the table above is that any second engine might do as
well, and the routing is decoration. That is worth answering with a control
rather than an argument, so the benchmark runs fixed pairings: web search plus
the *same* second engine for every question, whatever the question is about.

| configuration | answered | recall | searches/question |
|---|---|---|---|
| naive — web search only | 10/12 | 83% | 1.0 |
| routed to one engine, no web search | 10/12 | 83% | 1.0 |
| web + a fixed second engine (news) | 11/12 | 92% | 2.0 |
| web + a fixed second engine (scholar) | 11/12 | 92% | 2.0 |
| web + a fixed second engine (shopping) | 11/12 | 92% | 2.0 |
| **web + the routed second engine** | **12/12** | **100%** | **1.7** |
| routed to 3 engines | 12/12 | 100% | 2.0 |
| routed to 3 engines, 2 rounds | 12/12 | 100% | 3.9 |

All three fixed pairings miss the same single question, and it is the one that
structurally requires a particular engine: whether interest in something is
rising over time is answered by a demand series, and no quantity of news,
scholarly or shopping results substitutes for one.

So adding *a* second engine is worth nine points of recall. Adding the *right*
one is worth seventeen, and costs less — 1.7 searches per question against 2.0 —
because on the four questions where routing detects no signal it issues a single
search, while a fixed pairing pays for a second engine that had nothing to add.

That is the whole claim, stated as narrowly as the evidence supports it: routing
is not what adds the second engine, it is what decides which one and when not to
bother.

### What depth buys

Nothing, on this question set. A third engine and a second round cost 2.2 more
searches per question for identical answers. The defaults were 3 engines and 4
rounds until this was measured; they are now 2 and 1.

Twelve questions is a small set, and one whose answers are reasonably
discoverable, so a harder set may well pay for depth — which is why rounds stay
configurable and the stopping rule still governs them.

### Reproducing this

The recorded responses are committed, so this needs no API key and spends
nothing:

```bash
python -m frugal.benchmark --replay --cache benchmarks/fixtures
```

That should print the table above. The question set, including the reasoning
behind each question, is in [benchmarks/questions.json](benchmarks/questions.json).
Four of the twelve are ordinary factual questions that plain web search answers
perfectly well; they are there because a set the planner wins outright would be
a set chosen to make it win.

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
pytest                                                    # test suite
python -m frugal.benchmark --replay --cache benchmarks/fixtures
```

To run it live against SerpApi instead, check the cost first:

```bash
python -m frugal.benchmark --dry-run     # projected spend, issues nothing
python -m frugal.benchmark               # the real thing
```

## Which SerpApi engines are used

Frugal routes across engines rather than defaulting to web search: `google`,
`google_news`, `google_scholar`, `google_patents`, `google_shopping`,
`google_jobs`, `google_trends`, `google_maps`, `google_finance`. Each carries
cost, latency and volatility metadata that the planner reads when deciding
where a question should go and how long its answer stays fresh.

## Licence

MIT — see [LICENSE](LICENSE).
