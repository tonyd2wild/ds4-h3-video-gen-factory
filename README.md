# DS4 x H3 Video Gen Factory

![banner](docs/banner.png)

**Run DeepSeek-V4-Flash at full 1M context AND two instances of MiniMax H3 video
generation on the same two DGX Sparks. At the same time. Nothing turned off.**

Two NVIDIA DGX Sparks (GB10, 121 GiB unified memory each). DeepSeek-V4-Flash
serves agents at TP=2 across both boxes with a **1,473,052 token KV pool** and a
**1,048,576 token context window**. On top of that, two independent ComfyUI +
MiniMax H3 instances render 15-second 480p video with audio, one per node.

No `--reserve-vram`. No reduced KV. No shortened context. No restarts between
modes. You launch the language model, you launch the video instances, and they
coexist.

Everything below is measured on the hardware, not estimated. Where a number is
uncertain it says so.

---

## ⛳ Start here if you do not have DeepSeek-V4-Flash running yet

**This repo assumes DS4 is already deployed on your Sparks.** It is about running
video generation *alongside* it, not about getting the language model up in the
first place.

If you are starting from nothing, deploy DS4 first with the recipe this build
runs on:

### 👉 [**DeepSeek-v4-Flash-0731-DSpark 1M NVFP4-KV, 2x DGX Spark**](https://github.com/tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark)

That is the exact configuration benchmarked here: the 0731 build, DSpark
speculative decoding, NVFP4 MLA KV cache, TP=2 across two Sparks. Get that
serving and returning tokens, then come back here and add the video layer.

Other DS4 variants (500K ctx, 900K ctx, abliterated) are on the
[same profile](https://github.com/tonyd2wild?tab=repositories) if you want a
different context or memory tradeoff. The co-tenancy behaviour in this repo
should hold for any of them, but the numbers here were measured on 0731 1M.

---

## The result

Throughput of DeepSeek-V4-Flash while MiniMax H3 renders on the same nodes.
Concurrency C1 through C6 (C6 is the ceiling, `--max-num-seqs 6`).

| Concurrency | Idle (no video) | 1 render | 2 renders |
|---:|---:|---:|---:|
| C1 | 88.87 | 40.98 | 28.48 |
| C2 | 149.37 | 68.38 | 50.99 |
| C3 | 199.47 | 88.19 | 66.74 |
| C4 | 214.90 | 97.19 | 73.44 |
| C5 | 203.93 | 92.14 | 74.25 |
| C6 | **285.95** | **130.77** | **100.79** |

*Aggregate tokens/sec across all concurrent streams.*

**You keep 46% of throughput with one render running, and 35% with two.**

### The second render is nearly free

This is the part worth noticing. At C1, going from no video to one video costs
you 47.9 tok/s. Going from one video to two costs you **12.5**. At C6 the first
render costs 155 tok/s and the second costs 30.

The first video render takes the contention hit. The second one largely rides
along on capacity the first already gave up.

### Latency never breaks

| | C1 TTFT | C6 TTFT |
|---|---:|---:|
| Idle | 0.160 s | 0.338 s |
| 1 render | 0.315 s | 0.804 s |
| 2 renders | 0.428 s | 0.882 s |

Time-to-first-token stays under a second in every condition. **The endpoint
always answers immediately.** It generates more slowly while the GPUs are busy,
but it never queues, stalls, or drops a request. Every sweep completed 6/6.

And when the renders finish, full speed returns on its own. Nothing to restart.

### Why that matters

You can run a **video generation loop** and keep using your agents. An agent that
prompts between renders never sees the penalty at all. An agent that prompts
during one gets a slower answer, not a failed one.

---

## ⚠️ Read this before comparing our numbers to yours

**On a speculative-decoding model, throughput is a function of your prompt, not
just your hardware.** DeepSeek-V4-Flash here runs `dspark` speculative decoding
with 5 draft tokens. The drafter proposes, the target model accepts or rejects.
Predictable output means more accepted tokens per forward pass, which means more
throughput, on identical hardware.

Measured on this box, same model, same minute, only the prompt changed:

| Prompt type | tok/s | accepted tokens per pass |
|---|---:|---:|
| Dense technical prose | 35.9 | 2.45 |
| Agent-style tool reasoning | 57.6 | 3.82 |
| Code generation | 64.2 | 4.40 |
| Structured JSON output | 67.3 | 4.65 |
| **Counting 1 to 300** | **91.6** | **5.94** |

**A 2.6x spread from the words alone.**

Every benchmark in this repo uses the counting prompt, greedy (`temperature 0`),
because it is the **ceiling**. That is deliberate: we are measuring what a video
render *costs*, so both sides of the comparison need to sit at the same point on
that curve. If you benchmark with prose you will get roughly 40% of these
numbers, and that is your prompt, not your machine.

Quote the prompt class alongside any tok/s figure or the figure means nothing.

### The measurement bug you will probably hit

Speculative decoding emits **every accepted token in a single SSE chunk**. If you
count streaming chunks and call them tokens, you undercount by the acceptance
length. We measured 200 real tokens arriving in 81 chunks: a **2.47x**
understatement.

Ask the server for its own count:

```json
"stream": true,
"stream_options": {"include_usage": true}
```

and read `completion_tokens` from the final chunk. `bench/bench_conc.py` does
this and cross-checks against a non-streaming call.

---

## Hardware

| | |
|---|---|
| Nodes | 2x NVIDIA DGX Spark (GB10) |
| Memory | 121 GiB unified per node (CPU and GPU share one pool) |
| Interconnect | 200G RoCE, ConnectX-7, `rocep1s0f0` |
| Fabric | 192.168.x.0/24 private, both nodes |

**Unified memory is the whole story on this platform.** There is no separate
VRAM. Everything below competes for the same 121 GiB, and page cache counts.

---

## The memory budget

Per node, measured:

| Component | GiB |
|---|---:|
| DS4 weights (TP=2 half-share) | **79.51** |
| KV cache @ `--gpu-memory-utilization 0.78` | 10.28 |
| Activations + CUDA graph capture | ~5.2 |
| OS, docker, container overhead | ~10 |
| **DS4 total** | **~105** |
| **Free for video** | **~16-18** |

MiniMax H3 idle sits near 6 GiB. Rendering 480p it fits inside that headroom.

### The trap: H3's footprint is not fixed

H3 loads its components one at a time (audio VAE 576 MB, video VAE 4,965 MB,
text encoder 14,956 MB, DiT 19,995 MB). What it *holds* depends on what the box
lets it hold.

- On a 24 GB RTX 3090, it evicts as it goes and peaks at **~14.6 GB**.
- On an empty 121 GiB Spark, nothing forces eviction, so it keeps everything
  resident and peaks at **~50 GB**.
- With DS4 already holding 105 GiB, it is squeezed back down and runs inside the
  remaining headroom.

Same model, same workflow, 3x the footprint, purely because of available memory.
**Do not plan capacity from an idle measurement or from a measurement taken on an
empty box.**

**This is exactly why start order matters.** H3 is not greedy by design, it is
adaptive: it takes what is there. So whoever loads first defines the split. Load
DS4 first and it sets a hard floor that H3 then works within. Load H3 first onto
an empty node and it will take 50 GB simply because 50 GB was available, and DS4
will not fit afterwards.

### Utilization is a cliff, not a slope

We swept it. `--gpu-memory-utilization` is a single budget covering weights,
activations, CUDA graphs and KV. Weights are fixed at 79.51 GiB, so cuts land
almost entirely on KV.

| util | KV cache | Result |
|---:|---:|---|
| 0.78 | 10.28 GiB (1,473,052 tok) | Full 1M context. **Use this.** |
| 0.70 | 1.14 GiB | Boots, but vLLM computes max usable context at **2,816 tokens**. Useless. |
| 0.68 | 0 blocks | `ValueError: No available memory for the cache blocks`. Will not start. |

**Counter-intuitive but important: do not lower utilization to make room for
video.** Between 0.78 and 0.70 you free about 12 GiB and destroy your context
window to do it. Leave it at 0.78 and let the video model live in the headroom
that is already there. We tried it the other way first and it was wrong.

---

## Deploy

### 🚨 Start order is not a preference. It is the whole trick.

**Launch DeepSeek-V4-Flash FIRST. Then launch MiniMax H3.**

This is the single most important line in this repo. Reverse it and the setup
does not work.

H3 sizes itself to the memory that is free when it loads. Start it on an idle
121 GiB node and it will keep every component resident and take **~50 GB**. DS4
then needs ~105 GiB and cannot get it, so the language model fails to start.

Start DS4 first and it claims its ~105 GiB up front. H3 then loads into the
16-18 GiB that remains, evicts components as it goes exactly like it does on a
24 GB consumer card, and both run.

Same two programs, same two boxes. One order works and the other does not, and
nothing in either program's output will tell you that is why.

The corollary: **if you restart DS4, stop the H3 instances first.** Otherwise
they are holding the memory DS4 needs to come back up.

### The scripts

See [`deploy/`](deploy/). **Prerequisite: DS4 deployed via the
[0731 DSpark recipe](https://github.com/tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark).**

1. [`deploy/launch_ds4_pair.sh`](deploy/launch_ds4_pair.sh) — DS4 at TP=2 across
   both nodes. **Run this first and let it finish loading** (watch for
   `GPU KV cache size: 1,473,052 tokens`).
2. [`deploy/launch_h3_instance.sh`](deploy/launch_h3_instance.sh) — one ComfyUI +
   H3 instance. Run once per node, **only after step 1 is serving**.

### Non-obvious things that will cost you an evening

**`--privileged` and `--device /dev/infiniband` are required on the DS4
containers.** Without the IB devices inside the container, NCCL cannot load an IB
net plugin. Because the recipe forces `NCCL_NET=IB` there is no ethernet
fallback, so it dies with `NCCL error: invalid usage` and a
`Failed to initialize any NET plugin` warning that points at the network instead
of at the missing device. The fabric will test perfectly fine while this happens.

**Never bind a launch script from `/tmp`.** `/tmp` is cleared on reboot. Docker
then recreates the missing bind source as a *directory*, and the container exits
127 while `docker ps` still reports it as `Up`. Stage scripts in `/var/tmp`.

**The ComfyUI image entrypoint is not python.** You must pass
`--entrypoint /opt/env/bin/python3` explicitly or the container runs `sleep` with
your arguments and exits 1.

**Drop page caches before launching.** These are unified-memory boxes. Page cache
from a previous model load or a video write starves the GPU allocator, and you
get `NVRM: ... NV_ERR_NO_MEMORY` in `dmesg` followed by a worker dying mid-warmup.
`sync; echo 3 > /proc/sys/vm/drop_caches` on every node first.

**Start the worker before the head.** If the head restarts while the worker holds
loaded weights, they desync: the worker sits at ~90 GiB waiting and the head
never rendezvous. You will see `TCPStore ... Broken pipe` on the worker.

**`--max-mamba-cache-size` interacts with concurrency.** vLLM caps running
requests at `mamba_cache_size / 3`.

---

## Video generation

MiniMax H3 reference-to-video, via ComfyUI's API. See
[`workflows/`](workflows/) for working graph builders.

| Constraint | Value |
|---|---|
| Frame count | must be `17n + 5`. 362 frames = 15.08 s @ 24 fps |
| Dimensions | divisible by 32. 832x480, 960x544, 1280x720 |
| Sampler | `res_multistep` via `BasicGuider` + `SamplerCustomAdvanced` |
| Steps | 20 |

**`KSamplerSelect` alone will not work** — H3 needs the
`BasicGuider` + `SamplerCustomAdvanced` pair. A plain `KSampler` raises
`IndexError`.

**Audio decodes from the sampler latent**, not from a third output of
`MiniMaxH3ReferenceToVideo`. That node returns only `CONDITIONING` and `LATENT`.
Wire `VAEDecodeAudio.samples` to the `SamplerCustomAdvanced` output, same as the
video decode. Getting this wrong yields
`Exception when validating inner node: list index out of range`.

**Launch ComfyUI with `--disable-pinned-memory`.** ComfyUI page-locks most of
available RAM by default, which on a unified-memory box takes it away from
everything else.

**Every reference image slows every sampling step**, not just the first. It is
not a fixed encode cost. Use the references you need for identity consistency and
no more.

---

## Reproduce

```bash
# on the node, not over the network — network latency contaminates TTFT
python3 bench/bench_conc.py 127.0.0.1:8888 <model-name> "label" 1,2,3,4,5,6
```

Stdlib only. Streams every request for real TTFT, measures decode
first-token-to-last-token so prefill is not smeared into throughput, gives each
concurrent stream a different counting range so no two share a prefix-cache
entry, and takes token counts from the server's `usage` block.

Raw output in [`bench/results/`](bench/results/).

---

## Render wall-clock

Two clips, same settings, same 480p, same length, both rendered while DS4 was
loaded and serving on the same nodes.

| Clip | Conditioning | Total | Sampling | Per step |
|---|---|---:|---:|---:|
| A | 1 reference image | 28:31 | 27:00 | 81.03 s |
| B | 6 reference images | 28:54 | 26:16 | 78.82 s |
| C | 1 reference **video** (15 s) + 2 images | 23:48 | — | — |

832x480, 362 frames (15.08 s @ 24 fps), 20 steps, `res_multistep`, audio on.

### Reference images are effectively free

Six references cost 23 seconds more than one, and were actually *faster* per
step. Whatever difference exists is inside the run-to-run noise at this
resolution and length.

That is worth knowing because identity consistency is the main reason to add
references, and it turns out you are not paying for it per step. Clip B carried
six: a character sheet each for three people, plus a vehicle and two locations.
Nobody drifted between cuts and it cost nothing measurable.

The up-front cost is real but small: references are encoded once before sampling
starts, which is why a multi-reference render can sit at step 0 for a while and
look stuck when it is not.

Clip C points the same way. It was conditioned on an entire 15-second video plus
two character sheets, which is by far the heaviest conditioning of the three, and
it did not take longer. `MiniMaxH3ReferenceToVideo` accepts `ref_videos`
alongside `ref_images`, and feeding it the previous clip produced a genuine
continuation: same room, same light, same performer, rather than a fresh reading
of a text description.

**One caveat on clip C specifically.** It also ran with less competing traffic
than A and B, which were rendering during the concurrency sweeps. So its 23:48
is not evidence that a video reference makes things *faster*. The defensible
claim is narrower and still useful: **adding a full video reference did not make
it slower.**

### ⚠️ What these timings are and are not

**DS4 was loaded and serving during both renders, but not under sustained load.**
The concurrency sweeps in this repo ran against it during parts of both, so both
clips saw real traffic in bursts, not continuously.

So treat 28:31 and 28:54 as a **light-to-moderate load** figure. They are not a
clean idle baseline, and they are not what you would get with an agent fleet
hammering the endpoint for the full render.

**The reverse direction is still unmeasured.** Everything above is what video
costs the language model. What sustained language-model traffic costs *render
wall-clock* has not been tested, and we are not going to guess at it.

---

## Full write-up

The longer version, including the two wrong turns and why they were wrong:
[**docs/ARTICLE.md**](docs/ARTICLE.md)

---

## Honest limitations

- **One hardware configuration.** Two GB10 Sparks. Not validated anywhere else.
- **480p video.** 720p renders peaked at ~50 GiB on an unloaded box. Whether it
  fits alongside DS4 under memory pressure is untested. It may well work, since
  H3 demonstrably shrinks under pressure, but we have not run it, so we are not
  claiming it.
- **We did not measure the reverse direction.** These numbers are what the video
  costs the language model. What agent traffic costs *render wall-clock* is still
  open.
- **Short generations.** 700 max tokens per stream. Long-context behaviour under
  co-tenancy is unmeasured.
- **`dmesg` shows NVRM out-of-memory events** on the co-tenanted node. Nothing
  died across our runs and both services stayed healthy, but headroom is genuinely
  thin and we are reporting it rather than hiding it.

---

## Credits

Built and measured by [@tonyd2wild](https://x.com/tonyd2wild).

Standing on: DeepSeek-V4-Flash, MiniMax H3, vLLM, ComfyUI, and the DGX Spark
community recipes that got DS4 onto GB10 in the first place.

MIT.
