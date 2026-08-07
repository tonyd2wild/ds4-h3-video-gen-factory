# Two MiniMax H3 instances on two DGX Sparks, with a 1M-context LLM still serving. Here's what it costs.

**MiniMax H3** is the reason for this post. Two separate instances of it, one per
node, rendering 15-second 480p clips with synchronised audio on two NVIDIA DGX
Sparks.

The part worth writing down is what else was running at the same time. Those same
two Sparks were serving **DeepSeek-V4-Flash** at full 1,048,576-token context
with a 1,473,052-token KV cache, answering my agents the entire time both renders
were in flight.

Nothing turned off. No reduced context. No shrunken cache. No restarting between
"LLM mode" and "video mode."

We expected this to be impossible. We were wrong twice, in both directions, and
the wrong turns are more instructive than the result.

---

## The number

Aggregate throughput of DeepSeek-V4-Flash across concurrent streams, C1 to C6
(C6 is the ceiling, `--max-num-seqs 6`):

| Concurrency | Idle | 1 render | 2 renders |
|---:|---:|---:|---:|
| C1 | 88.87 | 40.98 | 28.48 |
| C2 | 149.37 | 68.38 | 50.99 |
| C3 | 199.47 | 88.19 | 66.74 |
| C4 | 214.90 | 97.19 | 73.44 |
| C5 | 203.93 | 92.14 | 74.25 |
| C6 | **285.95** | **130.77** | **100.79** |

You keep 46% of throughput with one render going, 35% with two.

**The second render is nearly free.** At C1, going from zero renders to one costs
47.9 tok/s. Going from one to two costs 12.5. At C6 the first costs 155 and the
second costs 30. The first video render absorbs the contention; the second one
largely rides on capacity the first already surrendered.

Time-to-first-token never breaks a second in any condition: 0.16s idle, 0.32s
with one render, 0.43s with two at C1. Even at C6 with both rendering it is
0.88s. Every sweep completed 6 of 6. Nothing queued, stalled, or dropped.

When the renders finish, full speed returns by itself. Nothing to restart.

**That is the practical point.** You can run a video generation loop and keep
using your agents. An agent that prompts between renders never sees the penalty.
An agent that prompts during one gets a slower answer, not a failure.

---

## Wrong turn one: we assumed the video model had a fixed size

MiniMax H3 loads its components sequentially: audio VAE at 576 MB, video VAE at
4,965 MB, text encoder at 14,956 MB, and the DiT at 19,995 MB.

On a 24 GB RTX 3090, memory pressure forces eviction as it goes, and the process
peaks around **14.6 GB**. So we reasoned that the DiT is the binding constraint
and budgeted ~21 GB.

Then we measured it on an empty 121 GiB Spark and watched it climb to **50 GB**.

Nothing was leaking. With that much memory available, nothing forced eviction, so
ComfyUI simply kept every component resident. Same model, same workflow, three
times the footprint, entirely because the box permitted it.

We then declared 720p video and the LLM couldn't share a node, at any setting.
That was also wrong, and for the same reason inverted: put DS4 on the box first,
and H3 gets squeezed back down and runs inside whatever headroom remains, exactly
as it does on a 3090.

**A memory measurement describes the conditions it was taken under, not the
requirements of the workload.** Ask what the box *allowed* versus what the job
*needed*. On unified memory those are very different questions.

### Which is why start order is the whole trick

H3 is not greedy by design. It is adaptive: it takes what is free when it loads.
So **whoever starts first defines the split.**

Start DS4 first and it claims ~105 GiB up front, leaving 16 to 18. H3 then loads
into that, evicting components as it goes exactly as it does on a 24 GB consumer
card, and both run.

Start H3 first on an idle node and it will take ~50 GB, simply because 50 GB was
there. DS4 then needs 105 and cannot get it, and the language model never starts.

Same two programs, same two boxes, and nothing in either program's output tells
you the order was the problem. If you restart DS4 later, stop the H3 instances
first for the same reason.

---

## Wrong turn two: we tuned the wrong knob

The intuitive move to make room for video is to lower vLLM's
`--gpu-memory-utilization`. It is a single budget covering weights, activations,
CUDA graph capture and KV cache. Free some, hand it to the renderer.

We swept it:

| util | KV cache | Outcome |
|---:|---:|---|
| 0.78 | 10.28 GiB (1,473,052 tok) | Full 1M context |
| 0.70 | 1.14 GiB | Boots. vLLM computes usable context at **2,816 tokens** |
| 0.68 | 0 blocks | `ValueError: No available memory for the cache blocks` |

Model weights are fixed at 79.51 GiB per node and do not move. So every gigabyte
you reclaim comes almost entirely out of KV. Between 0.78 and 0.70 you free about
12 GiB and destroy your context window doing it. Below that it will not start at
all.

**Leave utilization at 0.78 and let the video model live in the headroom that is
already there.** At 0.78, DS4 uses about 105 GiB of 121 and leaves 16 to 18 free.
480p H3 fits in that. We spent an hour engineering around a constraint that the
default configuration did not have.

There is a smaller lesson in there too: when vLLM refused to start at 0.68 it
suggested *raising* utilization. That is correct advice in isolation and
completely wrong for what we were trying to do. Error messages optimize for the
common case.

---

## The benchmark is a function of your prompt

This one is not about co-tenancy at all, and it is the finding most likely to
matter to anyone comparing numbers.

DeepSeek-V4-Flash runs `dspark` speculative decoding with 5 draft tokens. A small
drafter proposes a run of tokens, the target model verifies them in one pass, and
accepted tokens come free. How many get accepted depends on how predictable the
output is.

Same box, same model, same minute, only the prompt changed:

| Prompt type | tok/s | accepted tokens per pass |
|---|---:|---:|
| Dense technical prose | 35.9 | 2.45 |
| Agent-style tool reasoning | 57.6 | 3.82 |
| Code generation | 64.2 | 4.40 |
| Structured JSON | 67.3 | 4.65 |
| Counting 1 to 300 | **91.6** | **5.94** |

**A 2.6x spread from the words alone.**

We benchmarked with explanation prompts first and reported ~40 tok/s as the
machine's speed. It is the machine's speed on the least predictable content
available. Counting is the ceiling.

Every benchmark here uses the counting prompt at `temperature 0`, deliberately,
because we are measuring what a video render *costs* and both sides of that
comparison need to sit at the same point on the acceptance curve. If you bench
with prose you will get roughly 40% of these numbers, and that is your prompt,
not your hardware.

**Quote the prompt class alongside any tok/s figure on a speculative-decoding
model, or the figure is not meaningful.**

### The measurement bug underneath it

Speculative decoding emits every accepted token in a **single** SSE chunk. Count
streaming chunks and call them tokens, and you undercount by the acceptance
length. We measured 200 real tokens arriving in 81 chunks: a **2.47x**
understatement, which is exactly the gap that made our first numbers look absurd.

Ask the server for its own count:

```json
"stream": true,
"stream_options": {"include_usage": true}
```

then read `completion_tokens` from the final chunk, and cross-check against a
non-streaming call. Do not count chunks.

---

## Four traps that cost real time

**NCCL needs the InfiniBand devices inside the container.** DS4 containers must
run `--privileged` with `--device /dev/infiniband`. Without them NCCL cannot load
an IB net plugin, and because the recipe pins `NCCL_NET=IB` there is no ethernet
fallback. It dies with `NCCL error: invalid usage` and a
`Failed to initialize any NET plugin` warning. Both point at the network. The
fabric will test perfectly healthy while this happens. We diffed all 66
environment variables against a working node before noticing the container flags.

**Never bind a launch script from `/tmp`.** It is cleared on reboot. Docker then
recreates the missing bind source as a *directory*, and the container exits 127
while `docker ps` still reports it as `Up`. Stage in `/var/tmp`.

**Drop page caches before launching anything.** On unified memory the page cache
competes with the GPU allocator. Leftover cache from a model load or a video
write produces `NVRM: ... NV_ERR_NO_MEMORY` in `dmesg` and a worker that dies
mid-warmup, which reads as a mystery crash.

**Start the worker before the head.** If the head restarts while the worker holds
loaded weights they desync: the worker sits at ~90 GiB waiting for a peer that
never arrives, and you see `TCPStore ... Broken pipe`. One node's memory climbs
and the other's does not, which is a useful signature.

Also, minor but wasteful: the ComfyUI container's default entrypoint is not
python. Without `--entrypoint /opt/env/bin/python3` it runs `sleep` with your
arguments and exits 1.

---

## What we did not measure

- **One hardware configuration.** Two GB10 Sparks. Nothing else validated.
- **480p only.** 720p peaked at ~50 GiB on an unloaded box. Whether it fits
  alongside DS4 under pressure is untested. It plausibly does, given H3
  demonstrably shrinks when squeezed, but we have not run it so we are not
  claiming it.
- **The reverse direction.** These numbers are what video costs the LLM. What
  agent traffic costs render wall-clock is still open.
- **Long generations.** 700 tokens per stream. Long-context behaviour under
  co-tenancy is unmeasured.
- `dmesg` shows NVRM out-of-memory events on the co-tenanted node. Nothing died
  across our runs and both services stayed healthy, but headroom is genuinely
  thin and it would be dishonest to omit that.

---

## If you want to run this yourself

**Start with the language model.** This is about adding video *to* a working DS4
deployment, not about standing DS4 up. The recipe this was built on, and the
exact configuration benchmarked above:

**[DeepSeek-v4-Flash-0731-DSpark 1M NVFP4-KV, 2x DGX Spark](https://github.com/tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark)**

Get that serving tokens first. Then add the video layer from the factory repo.

## Reproduce it

Everything is in the repo: the harness, raw benchmark output, both deploy
scripts, and working H3 workflow builders.

**github.com/tonyd2wild/ds4-h3-video-gen-factory**

Run the benchmark on the node, not across the network. Network latency
contaminates TTFT and the decode window.

```bash
python3 bench/bench_conc.py 127.0.0.1:8888 <model> "label" 1,2,3,4,5,6
```

Stdlib only. Streams for real TTFT, measures decode first-token to last-token so
prefill is not smeared into throughput, gives every concurrent stream a different
counting range so no two share a prefix-cache entry, and takes token counts from
the server.

---

*Built and measured by [@tonyd2wild](https://x.com/tonyd2wild). MIT.*
