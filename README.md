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

Thirty questions with verifiable answers, scored on whether the retrieved
evidence contains the answer. Scoring is exact string matching against marker
terms, so there is no judge model and nothing to take on trust.

| strategy | what it does | answered | recall | searches/question |
|---|---|---|---|---|
| naive | sends the question verbatim to web search | 28/30 | 93% | 1.0 |
| keyword | one web search, question reformulated | 29/30 | 97% | 1.0 |
| **planned** | routed across engines, under a budget | **30/30** | **100%** | 1.6 |

Three strategies rather than two, because two could not tell the mechanisms
apart. An earlier version compared only the first and the last, and credited
routing with work that reformulation was doing.

### Which mechanism earns what

| mechanism | questions it fixes | questions it breaks | cost |
|---|---|---|---|
| reformulation | `hospitals-coimbatore`, `python-jobs-chennai` | `crispr-paper` | none — still one search |
| routing | `crispr-paper` | — | +0.6 searches per question |

Reformulation is worth more than it sounds, and the failure it fixes is worth
seeing. Asked *"Where are the major hospitals in Coimbatore?"* verbatim, web
search returned a music video called *Major*, a NASA project assessment, and a
TED talk. The same question as `major hospitals coimbatore` returned hospitals
in Coimbatore. Sending a natural-language sentence to a keyword index is a real
failure mode of real agents, and stripping the question words costs nothing.

It also *breaks* one question, which is why it is reported rather than
celebrated: the verbatim CRISPR question found the paper and the keyword version
did not. Routing recovers it by going to `google_scholar`.

So each mechanism is worth roughly one question in thirty. That is a small
effect, honestly measured, and stating it as anything larger would not survive a
reader who re-ran the benchmark.

### Modality: the one consistent difference

| strategy | answered in the right modality |
|---|---|
| naive | 26/30 |
| keyword | 26/30 |
| **planned** | **30/30** |

Four questions ask whether something is rising or falling. Both baselines answer
them in prose — an article asserting a direction. The planner answers them with
a demand series that can be plotted, dated and compared. Both count as answered;
only one can be checked.

This is the difference that does not shrink with sample size, because it is
structural rather than statistical: web search does not return a time series at
any page depth.

### Cost behaviour

The planner issues exactly one search on 11 of the 30 questions — the ones where
routing detects no signal. It does not spend more when there is nothing to gain,
which is where its 1.6 average comes from rather than 2.0.

### Is the routing doing the work?

The obvious objection is that any second engine would do and the routing is
decoration. That deserves a control rather than an argument, so the benchmark
runs fixed pairings: web search plus the *same* second engine for every
question, whatever it is about, at the same plan size as the routed
configuration.

| configuration | answered | recall | searches/question |
|---|---|---|---|
| naive — verbatim question | 28/30 | 93% | 1.0 |
| keyword — reformulated, one engine | 29/30 | 97% | 1.0 |
| routed to one engine, **no web search** | 25/30 | 83% | 1.0 |
| web + a fixed second engine (news) | 30/30 | 100% | 2.0 |
| web + a fixed second engine (scholar) | 30/30 | 100% | 2.0 |
| web + a fixed second engine (shopping) | 29/30 | 97% | 2.0 |
| **web + the routed second engine** | **30/30** | **100%** | **1.6** |

Three things follow, and only one of them flatters the router.

**Routing buys no extra recall over a well-chosen fixed pairing.** News and
scholar both reach 100%. If the claim were "routing finds answers a fixed
strategy cannot", this table would refute it.

**Routing buys cost.** Same recall as the fixed pairings for 1.6 searches per
question against 2.0 — a fifth cheaper — because on the 11 questions where it
detects no signal it issues a single search, while a fixed pairing always pays
for a second engine that had nothing to add.

**Dropping web search is much worse than adding to it.** Routing to the
specialised engine *alone* scores 83%, below the naive baseline. The specialised
index is narrower, not better; the pairing is what works. This is the most
useful negative result in the benchmark, because it is the configuration an
enthusiastic reading of the idea would suggest building.

So, stated as narrowly as the evidence allows:

> Routing reaches the same recall as a fixed multi-engine strategy while issuing
> about 20% fewer searches, by not adding an engine when the question does not
> call for one.

The three-engine and two-round sweeps are not re-run at thirty questions. Both
were measured at twelve, where neither bought any recall, and at thirty they
would cost more than the benchmark they are checking. `--skip-deep` omits them;
drop the flag to pay for them.

### Do later rounds know what earlier ones found?

Yes, and without a model. After each round, the results so far are mined for
vocabulary the question never contained, and the next round's queries carry it.
This is pseudo-relevance feedback, and it keeps the plan deterministic: the same
evidence yields the same terms, so a plan that adapts still reproduces exactly.

The clearest case is `isro-latest-mission`. The question asks about "the latest
ISRO mission launch" and does not name the mission. Round one's results do, and
round two asks about `eos` — the mission designation — which no rewording of the
question could have produced.

Morphological variants are excluded, because they are not new vocabulary. An
earlier version nominated "launches" for a question about a "launch", and
"crispr" for a question about "CRISPR-Cas9", which spends a search to ask what
was already asked.

**It does not improve recall on this benchmark**, and the default is therefore
still a single round:

| configuration | answered | searches | distinct evidence |
|---|---|---|---|
| one round | 10/10 | 18 | 147 |
| two rounds, adaptive | 10/10 | 35 | 256 |

Measured on the first ten questions. Nearly twice the searches for three-quarters
more evidence and not one additional answer. The mechanism is real and the
evidence it adds is real; on questions this set contains, it is not worth the
money. Raise `max_rounds` for a harder set.

### What depth buys

Nothing, on this question set. A third engine and a second round cost 2.2 more
searches per question for identical answers. The defaults were 3 engines and 4
rounds until this was measured; they are now 2 and 1.

### Limitations

Worth reading before drawing conclusions from the tables above.

- **The effect is small.** Two questions out of thirty separate the planner from
  the verbatim baseline, and one separates it from the reformulated one. At this
  sample size neither gap is statistically meaningful on its own. The modality
  result is the sturdier one: four out of four, and structural rather than
  statistical.
- **The set has saturated on recall.** Twenty-eight of thirty fall to a single
  web search, and every two-engine configuration reaches 100%, so the benchmark
  can no longer distinguish routing from any-second-engine on quality. Only the
  cost difference is still measurable, and a harder set would be needed to say
  more.
- **Routing is lexical.** Three failure modes of that are handled — a signal
  word inside a name ("Steve Jobs biography"), a signal shortly after a negation
  ("not recent news"), and ambiguous words admitted only in unambiguous phrasings
  ("work as", never bare "work"). Others certainly remain: a synonym nobody
  listed will not fire, and a question whose meaning turns on syntax rather than
  vocabulary will be read wrongly.
- **Google News is partitioned by country edition**, and place detection is
  lexical, so a question naming an Indian subject without naming India queries
  the wrong edition and returns nothing. "What is the latest ISRO mission
  launch?" is the case in the set: `gl=in` returns a hundred results and the
  default edition returns none. One of the thirty questions still spends a search
  this way.
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

## Using it from an agent (MCP)

Most search tools an agent can reach hand back ten links and say nothing about
what that cost. The agent cannot tell whether a question was cheap, cannot cap
what it is willing to spend, and cannot find out beforehand. It just searches,
and someone gets the bill.

Frugal exposes an MCP server with the two tools that change that:

| tool | what it does |
|---|---|
| `frugal_plan` | which engines a question would reach, and the most it could cost — **issuing nothing** |
| `frugal_search` | runs the plan under a budget the caller sets, and reports what it spent alongside what it found |

```bash
pip install 'frugal[mcp]'
```

Then point any MCP client at it — Claude Desktop, Cursor, or your own agent:

```json
{
  "mcpServers": {
    "frugal": {
      "command": "frugal-mcp",
      "env": {
        "SERPAPI_API_KEY": "your-key-here"
      }
    }
  }
}
```

A search comes back with its cost attached:

```json
{
  "evidence": [ ... ],
  "cost": {
    "searches_billed": 2,
    "served_from_cache": 0,
    "steps_planned": 2,
    "steps_skipped_by_early_stop": 0
  },
  "engines_used": ["google", "google_trends"],
  "stopped_because": "completed the plan"
}
```

Every evidence item carries the engine and query that produced it, so an agent
can cite a claim rather than assert it. Budgets are capped server-side at 25
searches per call whatever the caller asks for: an agent looping on a tool is
the usual way a search bill becomes a surprise.

Set `FRUGAL_REPLAY=1` and `FRUGAL_CACHE_DIR=benchmarks/fixtures` to run the
server entirely from committed fixtures, with no key and no spend.

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
