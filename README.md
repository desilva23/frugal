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

The measured claim is narrower than the pitch: on the benchmark set, routing
reaches the same answers as a fixed multi-engine strategy while issuing about
15% fewer searches, and returns structured data where a web search returns prose.
It is measured rather than asserted, and anyone can re-run it from committed
fixtures without an API key — including the parts that did not go the way this
project wanted. See [Results](#results).

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
evidence contains the answer. Scoring is exact string matching against marker
terms, so there is no judge model and nothing to take on trust.

| strategy | answered | recall | searches/question |
|---|---|---|---|
| naive — one web search | 11/12 | 92% | 1.0 |
| planned | 12/12 | 100% | 1.7 |

### Is the routing doing the work?

The obvious objection is that any second engine would do as well and the routing
is decoration. That deserves a control rather than an argument, so the benchmark
runs fixed pairings: web search plus the *same* second engine for every question,
whatever it is about.

| configuration | answered | recall | searches/question |
|---|---|---|---|
| naive — web search only | 11/12 | 92% | 1.0 |
| routed to one engine, no web search | 10/12 | 83% | 1.0 |
| web + a fixed second engine (news) | 12/12 | 100% | 2.0 |
| web + a fixed second engine (scholar) | 12/12 | 100% | 2.0 |
| web + a fixed second engine (shopping) | 12/12 | 100% | 2.0 |
| **web + the routed second engine** | **12/12** | **100%** | **1.7** |
| routed to 3 engines | 12/12 | 100% | 2.0 |
| routed to 3 engines, 2 rounds | 12/12 | 100% | 3.9 |

**The fixed pairings reach 100% too.** Routing buys no additional recall here.
What it buys is cost: identical answers for 1.7 searches per question against
2.0, because on the questions where routing detects no signal it issues a single
search while a fixed pairing always pays for a second engine that had nothing to
add.

So the claim this benchmark supports, stated as narrowly as the evidence allows:

> Routing achieves the same recall as a fixed multi-engine strategy while
> issuing about 15% fewer searches, by not adding an engine when the question
> does not call for one.

An earlier version of this file claimed 83% against 100%, and that number was
wrong. The trend question required a `Series` object to count as answered, which
made it arithmetically impossible for a web-search baseline to score on it — web
search returns documents and never a series. The baseline's results plainly did
answer the question, carrying headlines such as "Why India is Seeing EV Interest
Rise". Removing that requirement moved the baseline from 10/12 to 11/12 and
moved the fixed pairings from 11/12 to 12/12, which is most of what the earlier
table was reporting.

### Modality

Recall is not the only thing that differs. Asked whether interest is rising,
the baseline returns prose and the planner returns a 53-point series:

| strategy | answered in the right modality | series returned |
|---|---|---|
| naive | 11/12 | 0 |
| planned | 12/12 | 1 |

Both answer the question. Only one can be plotted, compared across terms, or
inspected for when the change happened. This is reported beside recall rather
than folded into it, because folding it in is what produced the wrong number
above.

### What depth buys

Nothing, on this question set. A third engine and a second round cost 2.2 more
searches per question for identical answers. The defaults were 3 engines and 4
rounds until this was measured; they are now 2 and 1.

### Limitations

Worth reading before drawing conclusions from the tables above.

- **Twelve questions is too few**, and one question separates the strategies. At
  this size that is not a result.
- **The set is too easy and has saturated.** Eleven of twelve fall to a single
  web search, and every two-engine configuration reaches 100%, so the benchmark
  can no longer distinguish routing from any-second-engine on quality at all.
  Only the cost difference is still measurable.
- **Routing is lexical**, so it inherits the failure modes of lexical matching:
  "Steve Jobs biography" matches the employment signal, "where can I *work* as a
  Python engineer" does not, and a negation reads as an endorsement.
- **Deduplication compares URLs**, so one story syndicated across four outlets
  counts as four pieces of evidence. Marginal novelty overstates how much a
  batch actually added.
- **Place detection uses a curated list** weighted towards India. An unlisted
  town is not detected; the query still carries the name, so the engine is no
  worse off than it would have been.

### Reproducing this

The recorded responses are committed, so this needs no API key and spends
nothing:

```bash
python -m frugal.benchmark --replay --cache benchmarks/fixtures
python -m frugal.benchmark --replay --cache benchmarks/fixtures --ablate
```

The question set, including the reasoning behind each question, is in
[benchmarks/questions.json](benchmarks/questions.json). Four of the twelve are
ordinary factual questions that plain web search answers perfectly well; they
are there because a set the planner wins outright would be a set chosen to make
it win.

## Using it

```bash
frugal plan "Which companies are hiring Python developers in Chennai?"
```

Shows the routing scores, the query shaped for each engine, the parameters each
one receives, and what the whole thing would cost — without issuing anything.
Knowing the price before agreeing to pay it is the point.

```bash
frugal ask "Has search interest in electric vehicles in India been rising?"
```

Runs it. The cost counter ticks as searches are billed, cache hits are marked
free, and the stopping rule says why it stopped. A time series is drawn as a
sparkline, because it is the one kind of evidence a web search cannot return:

```
1. electric vehicles india
   ▁▁▁▁▁▁▃▄▄▄▅█▆▅▄▄▅▄▅▆▄▄▄▄▃▃▃▄▃▃▃▂▄▄▄▄▃▃▃▃▂▃▃▂▁▂▁▂▁▁▁▁▁
   53 observations, min 17, max 100, trending down (-6%)
```

Everything printed is the planner's own trace rather than decoration: the
routing scores are what selected the engines, the counter is the budget
governor, and the saturation line is the stopping rule explaining itself.

Add `--replay --cache benchmarks/fixtures` to run either command against the
committed fixtures, with no API key and no spend.

### Answering, rather than retrieving

```bash
frugal ask "..." --answer
```

Writes an answer from the evidence, citing it by number. This needs a
`GROQ_API_KEY` in your `.env` — a free one is at
[console.groq.com/keys](https://console.groq.com/keys) — and everything else
works without it.

It sits deliberately outside everything the benchmark measures. Routing,
reformulation, budgeting and the stopping rule are all deterministic so that the
result table reproduces without a key or a model, and a model anywhere in that
path would destroy the property. Synthesis runs after the plan has finished,
reads only what the plan retrieved, and changes no number in the results table.

Three things are enforced rather than requested. The model sees only the
retrieved evidence, so an answer invented from memory has nothing to cite.
Citations are checked against the evidence actually supplied, and invented ones
are stripped rather than shown as references a reader cannot follow. And an
answer that cites nothing is labelled as ungrounded, because an uncited answer
is either a refusal — which is fine — or the model's own memory, which is not.

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
