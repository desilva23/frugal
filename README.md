# Frugal

**A cost-aware search planner for SerpApi.**

One web search finds the answer to about four questions in five — and not the
same four twice. Recorded twice, a week apart, a single search found the answer
to the same question **both times for 65 of 100 questions**. A plan of two searches across two
different indexes did it for **92**.

![Questions answered against searches spent, in two recordings](docs/frontier.svg)

**The second search is the one worth buying: twelve or thirteen more answers in
both recordings. A third engine buys one, or none.** Knowing that before paying
for it is the point of the project.

Most agents search badly. They take a question, fire a single query at one
engine, take the top ten results, and stuff them into a context window. When
that fails they fire another query, and another, until something sticks or the
budget is gone — and nothing anywhere tells them what it cost.

Frugal treats retrieval as a planning problem. Given a question and an explicit
budget, it compiles a *search plan*: which of SerpApi's engines to call, with
which query shaped for each, under a ceiling the caller sets — then executes it
and reports what it spent.

Every figure here replays from two committed recordings with no API key and no
spend, **including the ones that went against the design.** Choosing the engine
per question does no better than always pairing web search with a broad index.
Rewording the query, and adding locale parameters to it, have no effect that
survives a second recording — and an earlier version of this file claimed both
did, from one. What reproducibly matters is one thing: asking a second,
different index. See [Results](#results).

## What an agent gets from this

Frugal ships as an integration, not just a library. It plugs into
[MCP](#using-it-from-an-agent-mcp), [LangChain](#langchain) and
[LlamaIndex](#llamaindex), and adds one thing none of them otherwise have:

**an agent can see what a search costs, and cap it.**

Every existing way an agent reaches SerpApi hands back results and says nothing
about the price. The agent cannot tell whether a question was cheap, cannot cap
what it is willing to spend, and cannot find out beforehand. It searches, and
someone gets the bill.

| tool | what it does |
|---|---|
| `frugal_plan` | which engines a question would reach and the most it could cost — **issuing nothing** |
| `frugal_search` | runs the plan under a caller-set budget, and reports what it actually spent |

Results come back with provenance attached, so a chain citing its sources can
name the search that produced a claim rather than asserting it.

**Two ceilings, because one was not enough.** Each call is capped at 25
searches, and the process as a whole is capped at 200 — configurable with
`FRUGAL_SESSION_BUDGET`. The per-call cap alone does not stop an agent in a
loop, which is the case it was written for: a hundred calls at twenty-five
apiece is two and a half thousand searches. Every response also carries the
session total, because a caller who can only see one call's cost cannot manage a
budget across many. Cache hits consume neither ceiling; the allowance bounds
money, and a cached search cost none.

## Why this exists

SerpApi bills per search. Every wasted call is real money, and naive agents
waste a great many of them:

- **Redundant searches.** The same query, reformulated slightly, returns
  substantially the same page. Nobody notices because nobody is counting.
- **Wrong engine.** A question about a paper goes to web search when
  `google_scholar` would have answered it in one call.
- **No stopping rule.** An agent that has already found the answer keeps
  searching, because nothing tells it the evidence has saturated.

Each of those is a planning failure, and each is measurable — and measuring
them did not go the way this list expects. Rewording really is redundant. Which
engine turned out not to matter measurably, only that there is a second one. And
the stopping rule never had occasion to fire. See [Results](#results).

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

One hundred questions with verifiable answers, scored on whether the retrieved
evidence contains the answer. Scoring is string matching against marker terms on
word boundaries, so there is no judge model and nothing to take on trust.

**Everything was recorded twice**, on 20 September and again on 27–30 September,
and both recordings are committed. Every number below replays from them without
an API key.

| strategy | first | second | found both times |
|---|---|---|---|
| one web search, the question verbatim | 79/100 | 83/100 | 65 |
| one web search, reduced to keywords | 73/100 | 86/100 | 63 |
| one web search, keywords + locale parameters | 87/100 | 84/100 | 74 |
| **the plan — two searches, two indexes** | **97/100** | **95/100** | **92** |

The last column is the one to read. A single web search answers about four
questions in five, but *which* four changes: **32 of the 100 verbatim answers
flipped between the two recordings.** Only 65 questions were answered both
times. The plan answered 92 both times, and there is no question it missed
twice.

### Why there are two recordings

The second one was not planned. Checking what a first-time user would see, the
most ordinary question in the set — *"Who invented the telephone?"* — returned a
tweet, a car review and the history of the Caesar salad. The recorded response
said Google had found **38** results. Re-run a week later, the same request
reported 393,000 and put Alexander Graham Bell second.

On 20 September, 92% of the Google responses in the first recording reported
fewer than a thousand results. A week later 6% did. The specialist engines were
unaffected. That looked like a broken baseline inflating the plan's lead, so
the baseline was recorded again.

It moved from 79 to 83. The alarming-looking result counts turned out to predict
almost nothing — responses reporting tiny counts answered as often as the rest.
What the second recording actually showed was larger and more ordinary: web
search results are not stable from one request to the next, and **a single
recording of a web benchmark is one sample, not a measurement.** An earlier
version of this file reported one recording's numbers to the question, and drew
conclusions from differences of one and two. Those are retracted below.

One thing about the second recording limits what it can show. The web step was
recorded again for every arm; **each plan's second engine was recorded once and
is shared by both.** The plan's two scores are therefore less independent than
the baseline's, and its steadiness is flattered by that. The gap between 65 and
92 is large enough to survive the caveat; it should not be read as exact.

### What survives a second recording, and what does not

| claim | first | second | verdict |
|---|---|---|---|
| a second search, from a different index, helps | +13 | +12 | **holds** |
| rewording the question into keywords changes recall | −6 | +3 | does not replicate |
| locale parameters on the web search change recall | +14 | −2 | does not replicate |
| a third engine helps | +1 | 0 | nothing measurable |
| choosing the engine beats always using Scholar | 0 | +2 | nothing measurable |

**One thing reproducibly matters: asking a second, different index.** It is
worth twelve or thirteen questions over the best single engine in both
recordings, and twelve to eighteen over a plain web search.

**Two claims this file used to make are withdrawn.** From the first recording
alone it reported that reformulation cost six questions and that engine
parameters earned fourteen, and named parameters the largest contributor. In the
second recording reformulation gained three and parameters lost two. The three
single-search arms cannot be told apart; their order reverses between
recordings. Neither number was ever a finding.

The parameters themselves are not in doubt where they are structural. Trends
without `geo` answers about the wrong country, jobs without `location` returns
vacancies from anywhere, and maps without `z` returns HTTP 400. Those are
correctness, established by the failures they fix, and they stay. What is
withdrawn is the claim that adding `gl` and `hl` to a *web* search buys recall.

### Web search is sent the question as asked

The plan used to reduce every query to keywords, web search included. It now
sends web search the question verbatim, and keeps the keyword forms for the
engines that match terms rather than parse sentences.

| the plan's web step | first | second | mean |
|---|---|---|---|
| keywords + parameters — the old default | 98 | 91 | 94.5 |
| **the verbatim question — the default now** | **97** | **95** | **96.0** |

The honest account of this decision is that it was made badly once. On the first
recording alone the verbatim form scored 97 against 98, and this file said the
change had been "tested" and "makes things worse". That was a one-question
difference on one recording. With two, the verbatim form averages higher and
varies less — and a difference of 1.5 is still inside the noise, so the
measurement does not decide it. What decides it is that *"invented telephone"*
is what returned the Caesar salad, and that parameters on the web step changed
no outcome in either recording: 98 and 91 with them, 98 and 91 without.

### Is the routing doing the work?

The obvious objection is that any second engine would do. That deserves a
control, so the benchmark runs fixed pairings: web search plus the *same* second
engine for every question, whatever the question is about.

| configuration | first | second | searches/question |
|---|---|---|---|
| routed to one engine, no web search | 84/100 | 83/100 | 1.0 |
| web + a fixed second engine (shopping) | 87/100 | 85/100 | 2.0 |
| web + a fixed second engine (news) | 95/100 | 93/100 | 2.0 |
| web + a fixed second engine (scholar) | 97/100 | 93/100 | 2.0 |
| **web + the routed second engine** | **97/100** | **95/100** | **2.0** |
| web + two routed engines | 98/100 | 95/100 | 2.3 |

Routing is never behind a fixed pairing and is ahead of the best one by two in
the second recording and level in the first. That is not a result. **On this
question set, choosing the engine per question has no measurable advantage over
always pairing web search with a broad index.**

What the control does show is that the second index has to be *broad*. Added to
a verbatim web search, shopping — a narrow index — is worth two to eight
questions; news and scholar are broad and are worth ten to eighteen. A signal in
the question selects the second engine on 72 of the 100, and that produces
better-shaped evidence — a demand series rather
than an article about one, a street address rather than a page mentioning a
city. It does not produce more answers.

For the 28 questions with no specialist signal — *"Who invented the telephone?"*
is not a places question or a jobs question — the router used to search once and
stop. It now adds a broad index. `bing` was tried for that role and is recorded
here because the result was surprising: asked *"Who created the periodic table of
elements?"*, with the request confirmed correct in the response's own
`bing_url`, it returned Cambridge Dictionary and Merriam-Webster entries for
"created", and the word "mendeleev" appeared nowhere in the payload.

### What more searches buy

| plan size | first | second | searches/question |
|---|---|---|---|
| one engine | 84/100 | 83/100 | 1.0 |
| **two engines** — the default | **97/100** | **95/100** | **2.0** |
| three engines | 98/100 | 95/100 | 2.3 |

The second search buys twelve or thirteen answers. The third buys one or none,
for 13% more searches. That is the whole curve: **one more search is worth
buying, and on this evidence nothing after it is.**

An earlier version of this file carried the curve further — 99 at 2.3 searches,
100 at 4.1, "the hundredth answer costs 184 searches by itself". Those were
single-recording differences of one question each, and the two-round arm they
came from adapts its second round to the first round's results, so it cannot be
replayed now that the web step has changed. The arm is gone from the sweep and
the claims with it. `--engines` and `--rounds` still exist; what they buy on a
harder question set is unmeasured.

### Questions a web page should not have answered

The hundred are questions web search can answer, which is why routing's choice
of engine does not distinguish itself on them. A separate, smaller set was
written to find questions where the specialist engine is the only one that can
serve: places with their street addresses, the organisations named on a patent
record, two search-interest series compared. Committed before being run, scored
by the same rules, reported separately rather than folded into the hundred, and
recorded once.

| strategy | answered | searches/question |
|---|---|---|
| **one web search, verbatim** | **8/8** | **1.0** |
| one web search, keywords | 5/8 | 1.0 |
| one web search, keywords + parameters | 7/8 | 1.0 |
| the plan | 8/8 | 2.0 |

**The premise was wrong.** Plain web search answered all eight, at half the
plan's cost. Indian directory sites publish hospital, pharmacy and ATM addresses
as ordinary web pages, so asking for a street address does not require a maps
index after all. Eight questions recorded once is far too few to conclude much,
and the set is kept because a benchmark that only contains cases the project
wins is not a benchmark.

### Which kind of evidence came back

| strategy | time series returned |
|---|---|
| any single web search | 0 |
| the plan | 12 |

Twelve questions ask whether something is rising or falling. A web search
answers them in prose and counts as answered; the plan also returns a demand
series that can be plotted, dated and compared. Asked whether interest in millets
exceeds quinoa, it returns both series — 37.0 against 50.0 — which is an answer
no web page publishes as data.

This is **stated as a fact about which engines were called, not scored as a
result.** Only `google_trends` produces a series and only the plan calls it, so a
score here would report the configuration rather than measure anything. An
earlier version did score it, out of thirty, and quoted the figure as the
project's sturdiest claim. It was not a claim at all.

### Do later rounds know what earlier ones found?

Yes, and without a model. After each round, the results so far are mined for
vocabulary the question never contained, and the next round's queries carry it.
This is pseudo-relevance feedback, and it keeps the plan deterministic: the same
evidence yields the same terms, so a plan that adapts still reproduces exactly.

The clearest case is `isro-latest-mission`. The question asks about "the latest
ISRO mission launch" and does not name the mission. Round one's results do, and
round two asks about `eos` — the mission designation — which no rewording of the
question could have produced.

The default is a single round, and what a second round is worth is not
currently measured: the arm that measured it could not be replayed after the web
step changed, and was removed rather than left quoting stale numbers.

### The budget and the stopping rule, shown firing

Neither fires anywhere in the benchmark, which is recorded under
[Limitations](#limitations). That is a fact about the question set rather than
about the code, and the difference is worth demonstrating rather than asserting,
so both are reproduced here against a transport that charges. This needs no key
and no network:

```bash
python scripts/demo_governors.py
```

**The budget refuses the search that would exceed it.**

```
plan wants     : 6 searches
budget given   : 2
issued         : 2        skipped: 4
SerpApi billed : 2
stopped        : budget exhausted
```

The third step is never issued and the remaining four are handed back. It does
not fire on the benchmark because the limit is 12 and a plan spends 2 — and it
*cannot* fire in replay at any limit, because replay serves from cache, a cache
hit is free, and an allowance that bounds money is not consumed by a search that
cost none. A replayed run at `--budget 1` executing its whole plan is that rule
working, not failing.

**The stopping rule stops when the next search would buy nothing.**

```
plan wants     : 3 searches
budget given   : 50        (deliberately not binding)
issued         : 2        skipped: 1
stopped        : stopped early: evidence saturated
novelty        : [0.00, 0.00]
```

The second batch returned nothing the first had not, so the third search was
never bought. It does not fire on the benchmark for a more interesting reason:
marginal novelty never falls. Each step queries a *different* index, and
Scholar's papers are not Google's pages, so nearly every batch is new even on
a question that was answered by the first search. The signal the rule
watches for does not occur in a cross-engine plan. It occurs when one index is
queried repeatedly, which is what it was written for.

Raising the threshold until it fired would manufacture the result rather than
measure it, so the threshold is unchanged and this paragraph exists instead.

### Limitations

Worth reading before drawing conclusions from the tables above.

- **Two recordings are two samples, not many.** A third of single-search answers
  changed between them, so any difference of a few questions between two arms in
  one recording means nothing, and this file has been caught treating it as
  though it did. The claims kept above are the ones with the same sign and a
  similar size in both. More recordings would tighten every number here.
- **The plan's second engine was recorded once.** Both recordings share it; only
  the web step was recorded twice. The plan's scores are therefore less
  independent than the baseline's and its steadiness is overstated by some
  amount this benchmark cannot put a number on.
- **The first recording's web results were strange.** 92% of its Google
  responses reported under a thousand results where a week later 6% did. It is
  kept as a recording rather than discarded as a bad one because its scores
  turned out close to the second's — but it was not a normal day, and a reader is
  entitled to weight it accordingly.
- **Local routing was reached by vocabulary, and a vocabulary is never
  finished.** The signal listed restaurants and cafes but not pharmacies,
  museums or coworking spaces, so six of the ten local questions never reached
  maps. Adding three words would only have moved the omission, so the signal now
  also matches shape — "where is X", "X located". On their own those were too
  broad: "Where is the bug in this regex?" went to maps. They now count only when
  the question also names somewhere, either a place the locale detector knows or
  a kind of institution — and an institution counts inside its own name, which
  the rule written to keep "Steve Jobs" from firing the jobs signal would
  otherwise discard. All ten local questions reach maps and no other question
  does. Irregular cases will still be missed; this is a lexical router and that
  is its ceiling.
- **A single result is a result.** Maps answers a query matching one specific
  place with a `place_results` object rather than a `local_results` list, and
  the adapter read only lists — so "Where is the Indian Institute of Science
  located?" scored as a miss while the record naming Bengaluru sat unread in the
  response. Both of those fixes together removed this file's one known false
  positive: `iitm-location` had been scoring on a sociology paper about caste
  that happened to contain "chennai" and "iit madras", and is now answered by a
  map record giving the campus address. The markers were never touched.
- **Routing's choice of engine earns nothing measurable here**, as the fixed
  pairings show in both recordings. It earns better-shaped evidence. A question
  set with flight prices, share prices and map pins would test it properly; this
  one does not have them.
- **Neither governor ever fires on this benchmark.** The budget and the stopping
  rule bound a run rather than improve it, and both are shown working above
  against a transport that charges — but in both recordings every plan reports
  "completed the plan". The budget is 12 and a plan spends 2. The stopping rule
  waits for new evidence to stop arriving, and it never does, because each step
  queries a *different* index and nearly every batch is new. The mechanism is
  sound for repeatedly querying one index; on a cross-engine plan its signal
  does not occur. Raising the threshold until it fired would manufacture the
  result rather than measure it. A circuit breaker is not judged by how often it
  trips, but nor has this one been seen to trip in use.
- **Scoring is lexical.** Markers match on word boundaries, which fixed the worst
  of it, but a correct answer phrased without any listed marker still scores as a
  miss, and a page mentioning a marker incidentally still scores as a hit.
- **The cost report is a lower bound.** A request that reaches SerpApi is a
  search SerpApi runs, whether or not the response arrives, so
  `searches_charged` undercounts by however much a run fails or retries. Over one
  session this project recorded 760 searches while the account counter showed
  817; in another, on a day of heavy timeouts, it sent 253 requests to record 140
  searches and was billed for 178. The client now reports requests sent alongside
  searches recorded, which is how the second figure is known at all.
- **Routing is lexical.** Three failure modes of that are handled — a signal word
  inside a name ("Steve Jobs biography"), a signal shortly after a negation ("not
  recent news"), and ambiguous words admitted only in unambiguous phrasings
  ("work as", never bare "work"). Others certainly remain.
- **Eight engines of SerpApi's hundred-odd are profiled.** Flights, hotels and
  finance are not, and those are precisely the questions where no web page has
  the answer — live prices. Their absence is why the routing control has so
  little to distinguish.
- **Google News is partitioned by country edition**, and place detection is
  lexical, so a question naming an Indian subject without naming India queries
  the wrong edition and returns nothing.
- **Deduplication compares URLs**, so one story syndicated across four outlets
  counts as four pieces of evidence.
- **Place detection uses a curated list** weighted towards India. An unlisted
  town is not detected; the query still carries the name, so the engine is no
  worse off.

### Reproducing this

Both recordings are committed, so this needs no API key and spends nothing:

```bash
# the second recording, 27–30 September
python -m frugal.benchmark --replay --cache benchmarks/fixtures
python -m frugal.benchmark --replay --cache benchmarks/fixtures --ablate

# the first recording, 20 September
python -m frugal.benchmark --replay --cache benchmarks/fixtures-first
python -m frugal.benchmark --replay --cache benchmarks/fixtures-first --ablate

# the eight structured questions, recorded once
python -m frugal.benchmark --replay --cache benchmarks/fixtures-first \
    --questions benchmarks/structured.json
```

Replay fails loudly on a missing fixture rather than skipping it, so each
committed set either covers the workload completely or says it does not.

**Nothing is written unless you pass `--out`**, so replaying to check a number
cannot change it. That used to be untrue: output defaulted to
`benchmarks/results.json` whatever the other flags said, and running the five
commands above in order left the committed results file holding the eight
structured questions. The committed results are `results.json` and
`ablation.json` for the second recording, `results-first.json` and
`ablation-first.json` for the first, and `recordings.json` for the comparison
between them — which `python scripts/compare_recordings.py` regenerates, and
which a test holds the headline figures in this file to.

The question sets, including the reasoning behind each question, are in
[benchmarks/questions.json](benchmarks/questions.json) and
[benchmarks/structured.json](benchmarks/structured.json). Twenty-five of the
hundred are ordinary factual questions that plain web search answers perfectly
well; they are there because a set the planner wins outright would be a set
chosen to make it win.

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

**It reports what it cost.** This was the one expense the tool was silent
about: searches are billed by SerpApi and counted everywhere, while the
synthesis call is billed by whoever serves the model and appeared in no total.
An answer now carries its token usage.

```
  1,380 tokens (1,200 prompt + 180 completion)
  — billed by the model provider, not by SerpApi
```

Tokens rather than money, deliberately. A price table would have to name a rate
per model, and a rate committed to a repository is stale the week the provider
changes it — a cost figure that is quietly wrong is worse than one the reader
converts themselves. An endpoint that reports no usage is recorded as unknown
rather than as zero, because a zero would read as free.

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

### LangChain

```bash
pip install 'frugal[langchain]'
```

A retriever that drops in where a vector store would go, and two tools an agent
can choose between:

```python
from frugal.langchain import build_retriever, frugal_tools

retriever = build_retriever(budget=6)
docs = retriever.invoke("Which companies are hiring Python developers in Chennai?")

docs[0].metadata["engine"]     # which engine produced this
docs[0].metadata["plan_cost"]  # what the retrieval spent

agent_tools = frugal_tools()   # frugal_plan and frugal_search
```

Every document carries the engine, query and rank that produced it, so a chain
citing its sources can name the search rather than assert the claim. The first
document carries the plan's cost, which a list of documents does not otherwise
tell you.

`frugal_plan` is the tool worth having: an agent can ask what a question would
cost before deciding to spend it. Both tools cap the budget server-side at 25
searches per call whatever the caller asks for.

### LlamaIndex

```bash
pip install 'frugal[llamaindex]'
```

```python
from frugal.llamaindex import build_retriever, frugal_tools

retriever = build_retriever(budget=6)
nodes = retriever.retrieve("Which companies are hiring Python developers in Chennai?")

nodes[0].node.metadata["engine"]     # which engine produced this
nodes[0].node.metadata["plan_cost"]  # what the retrieval spent

agent_tools = frugal_tools()         # frugal_plan and frugal_search
```

Same two shapes in LlamaIndex's containers, and the same cap — a test asserts
the three integrations share one budget ceiling, so a caller cannot route around
it by picking a framework.

One caveat worth stating: LlamaIndex expects a relevance score per node, and a
planner has no similarity to report — there is no embedding and no distance. The
score is reciprocal rank, which preserves the order the engines returned and is
honest about being ordinal. It is not a confidence.

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

Eight, out of the hundred-odd SerpApi offers: `google`, `google_news`,
`google_scholar`, `google_patents`, `google_shopping`, `google_jobs`,
`google_trends` and `google_maps`. Each carries cost, latency and volatility
metadata that the planner reads when deciding where a question should go and how
long its answer stays fresh. All eight bill one search per call, verified
against the account counter.

How often each is reached across the hundred questions:

| engine | questions |
|---|---|
| `google` | 99 |
| `google_scholar` | 41 |
| `google_trends` | 12 |
| `google_news` | 11 |
| `google_jobs` | 10 |
| `google_maps` | 10 |
| `google_shopping` | 9 |
| `google_patents` | 8 |

`google_scholar` is high because 28 of those are the fallback for questions with
no specialist signal, rather than a scholarly judgement.

**What is not served.** Flights, hotels and finance are not profiled, so a
question about a live fare, a room rate or a share price gets web search and
whatever it happens to carry. Those are the questions where routing would matter
most, since no web page holds the answer — which is a fair criticism of the
benchmark above, and is why it cannot show routing's engine choice earning
anything.

## Licence

MIT — see [LICENSE](LICENSE).
