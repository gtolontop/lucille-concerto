# lucille-concerto

A distillation floor you can watch. A grid of terminals streams a teacher
model's output while a judge scores each finished sample and the ones that
pass fill a dataset — so that a throughput figure like *"5 million tokens per
second"* stops being an abstraction and becomes something with a shape.

No build step, no dependencies. Open `index.html`.

## The idea

Every number on screen hangs off two choices, and nothing downstream is free:

```
racks × capacity per rack        →  tok/s the fleet can do
fleet tok/s × utilisation        →  tok/s it is doing right now
live tok/s ÷ tok/s per session   →  how many sessions are open
live tok/s ÷ tokens per sample   →  samples per second
live tok/s × judge pass rate     →  how fast the dataset fills
```

Pick 3 racks and 40 tok/s per session and you get ~50 000 concurrent
sessions. Pick 750 tok/s per session instead and the same silicon gives you
~900 sessions that each finish a sample in half a second. Same fleet, same
tokens, opposite shape — which is the whole point of looking at it.

## Controls

| | |
|---|---|
| **racks** | 1–5 NVL72 Rubin racks, or `max` (16) |
| **teacher** | `concerto-6` 1.6T-A55B · `glm-5.3` 744B-A40B · `glm-5.3-flash` 320B-A18B |
| **tok/s per session** | 10 → 750; sets how fast one panel reads |
| **on screen** | 1 → 128 panels; the rest of the fleet keeps running off-screen |
| **size** | target columns per panel — this is what sets the font size |

`[` `]` page through the sessions · click a panel to open it full screen,
`esc` to come back · `space` pauses · `h` hides the settings row · `r`
restarts the run with a new dataset seed.

## What is real and what is not

**Simulated:** the timing, and only the timing. Per-session speed jitters
around nominal, sessions wait for prefill, some sit queued behind a batch, a
few are stragglers on a long context, and fleet utilisation drifts. That is
what makes it look like a serving fleet instead of a metronome.

**Pre-written:** the code in the panels. 34 samples in Python, Rust, Go, C,
JavaScript and shell, replayed token by token. Nothing calls a model.

**Estimated:** the per-rack throughput. The derivation, its anchor and its
sources are in [docs/throughput.md](docs/throughput.md) — it is an estimate
with stated assumptions, not a benchmark, and it is capped by a compute
roofline so it can never claim more than the silicon could do.

**Extrapolated:** the fleet-level counters. A browser cannot run 250 000
sessions, so it runs a sample of them — up to 1024, fewer when each one is
fast — at the real per-session speed, and scales up. The sample size is
printed next to the page number. The dataset fills at the fleet's rate, not
the sample's, so a browser that drops frames cannot fake a slow run.

The `throttled` badge is about *this browser*, not the fleet: it means the
renderer is not keeping its own panels fed.

## Running it

Double-click `index.html`, or serve the folder if you prefer:

```sh
python -m http.server 8777
```

Then <http://127.0.0.1:8777>.

## Layout

```
index.html   markup, styling, the top bar
corpus.js    the 34 pre-written samples
floor.js     lexer, throughput model, simulation clock, canvas renderer
```

## Licence

MIT.
