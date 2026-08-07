# Two DGX Sparks, one 1M-context LLM, and two video models. All at once.

I run DeepSeek-V4-Flash across two DGX Sparks. Full 1,048,576-token context, a
1,473,052-token KV cache, serving my agents.

Tonight I put two MiniMax H3 video generators on the same two machines and
rendered 15-second clips with audio while the language model kept answering.

Nothing turned off. No reduced context. No shrunken cache. No restarting between
"LLM mode" and "video mode." I expected it not to work.

Here is what it actually costs.

---

## The numbers

DeepSeek-V4-Flash aggregate throughput, C1 through C6 concurrent streams. C6 is
the ceiling here (max-num-seqs 6).

**Idle, no video**
C1 88.9 · C2 149.4 · C3 199.5 · C4 214.9 · C5 203.9 · C6 285.9 tok/s

**One video rendering**
C1 41.0 · C2 68.4 · C3 88.2 · C4 97.2 · C5 92.1 · C6 130.8 tok/s

**Two videos rendering**
C1 28.5 · C2 51.0 · C3 66.7 · C4 73.4 · C5 74.3 · C6 100.8 tok/s

You keep 46% of throughput with one render running. 35% with two.

## The second render is nearly free

This is the part I did not expect.

Going from zero renders to one costs 47.9 tok/s at C1. Going from one to two
costs 12.5.

At C6 the first render costs you 155 tok/s. The second costs 30.

The first video absorbs the contention. The second largely rides on capacity the
first already gave up.

## Latency never breaks

Time to first token, idle to two renders:

C1: 0.16s → 0.32s → 0.43s
C6: 0.34s → 0.80s → 0.88s

Under a second in every condition. Every sweep completed 6 of 6. Nothing queued,
nothing stalled, nothing dropped.

The endpoint always answers immediately. It writes more slowly while the GPUs are
busy. And when a render finishes, full speed comes back on its own with no
restart and no flag change.

**So you can run a video generation loop and keep using your agents.** An agent
prompting between renders never sees the penalty. An agent prompting during one
gets a slower answer, not a failure.

---

## Start order is the whole trick

This is the single most important thing in this post.

**Launch the language model first. Then launch the video models.**

MiniMax H3 is not greedy. It is adaptive. It sizes itself to whatever memory is
free when it loads:

- On a 24 GB RTX 3090 it evicts components as it goes and peaks around 14.6 GB.
- On an empty 121 GiB Spark, nothing forces eviction, so it keeps everything
  resident and takes 50 GB.

Whoever loads first defines the split.

Start DS4 first and it claims its 105 GiB up front. H3 then loads into the 16 to
18 GiB that remains and behaves exactly like it does on a consumer card.

Start H3 first on an idle node and it takes 50 GB because 50 GB was there. DS4
then needs 105 and never starts.

Same two programs. Same two boxes. One order works and the other does not, and
nothing in either program's output tells you the order was the problem.

Corollary: if you restart the LLM later, stop the video instances first.

## Do not lower gpu-memory-utilization to make room

This was my first instinct and it is exactly backwards. I swept it:

- 0.78 → 10.28 GiB KV → full 1M context. Correct.
- 0.70 → 1.14 GiB KV → boots, but vLLM computes max usable context at 2,816
  tokens. Useless.
- 0.68 → zero cache blocks → refuses to start.

Model weights are fixed at 79.51 GiB per node and do not move. Every gigabyte you
claw back comes almost entirely out of KV. Between 0.78 and 0.70 you free about
12 GiB and destroy your context window doing it.

Leave it at 0.78. The video model lives in the headroom that is already there.

---

## Your benchmark is measuring your prompt

This one applies to anyone benchmarking a speculative-decoding model, and it is
the reason my first numbers were nonsense.

DS4 runs dspark speculative decoding with 5 draft tokens. A small drafter
proposes a run, the target model verifies it in one pass, accepted tokens come
free. How many get accepted depends on how predictable the output is.

Same box, same model, same minute, only the prompt changed:

- Dense technical prose: 35.9 tok/s, 2.45 accepted per pass
- Agent-style tool reasoning: 57.6 tok/s, 3.82
- Code generation: 64.2 tok/s, 4.40
- Structured JSON: 67.3 tok/s, 4.65
- Counting 1 to 300: 91.6 tok/s, 5.94

**A 2.6x spread from the words alone.**

Every number above uses the counting prompt at temperature 0, because it is the
ceiling and both sides of a comparison need to sit at the same point on that
curve. If you benchmark with prose you will get roughly 40% of these numbers, and
that is your prompt, not your hardware.

Quote the prompt class alongside any tok/s figure or the figure means nothing.

### The measurement bug you will hit

Speculative decoding emits every accepted token in a **single** SSE chunk. Count
chunks and call them tokens and you undercount by the acceptance length.

I measured 200 real tokens arriving in 81 chunks. A 2.47x understatement.

Set `stream_options: {include_usage: true}`, read `completion_tokens` from the
final chunk, and cross-check against a non-streaming call. Do not count chunks.

---

## Render times, and a surprise

Two clips, 832x480, 362 frames (15.08s at 24fps), 20 steps, audio on, both
rendered while DS4 was live on the same nodes.

- One reference image: 28:31 total, 81.03 s/step
- Six reference images: 28:54 total, 78.82 s/step

**Reference images are effectively free.** Six cost 23 seconds more than one and
were faster per step. That is inside the noise.

That matters because identity consistency is the entire reason to add
references. The six-reference clip carried character sheets for three people plus
a vehicle and two locations. Nobody drifted between cuts and it cost nothing
measurable.

References are encoded once, up front, before sampling starts. Which is why a
multi-reference render can sit at step 0 long enough to look wedged when it is
fine. I misread exactly that earlier tonight.

---

## Four traps that cost me real time

**NCCL needs the InfiniBand devices inside the container.** Run the LLM
containers `--privileged` with `--device /dev/infiniband`. Without them NCCL
cannot load an IB net plugin, and if your recipe pins `NCCL_NET=IB` there is no
ethernet fallback. It dies with `NCCL error: invalid usage`. Both that and the
accompanying warning point at the network. Your fabric will test perfectly
healthy while this happens. I diffed all 66 environment variables against a
working node before I thought to check the container flags.

**Never bind a launch script from /tmp.** It is cleared on reboot. Docker then
recreates the missing bind source as a directory, and the container exits 127
while `docker ps` still reports it as Up. Use /var/tmp.

**Drop page caches before launching.** On unified memory the page cache competes
with the GPU allocator. Leftover cache produces NVRM out-of-memory errors in
dmesg and a worker that dies mid-warmup, which reads as a mystery crash.

**Start the worker before the head.** If the head restarts while the worker holds
loaded weights they desync. The worker sits at 90 GiB waiting for a peer that
never arrives. One node's memory climbs and the other's does not.

---

## What I did not measure

- One hardware configuration. Two GB10 Sparks. Nothing else validated.
- 480p only. 720p peaked at 50 GiB on an unloaded box. Whether it fits alongside
  the LLM under pressure is untested. It plausibly does, since H3 demonstrably
  shrinks when squeezed, but I have not run it so I am not claiming it.
- **The reverse direction.** Everything here is what video costs the language
  model. What sustained agent traffic costs render wall-clock is still open.
- Both render times were measured with the LLM loaded and serving but under
  burst load, not sustained. Treat them as light-to-moderate, not as an idle
  baseline and not as a saturated one.
- dmesg shows NVRM out-of-memory events on the co-tenanted node. Nothing died
  across my runs and both services stayed healthy, but the headroom is genuinely
  thin and it would be dishonest to leave that out.

---

## Everything is in the repo

Harness, raw benchmark output, both deploy scripts, working video workflow
builders, and every trap above written down properly.

**github.com/tonyd2wild/ds4-h3-video-gen-factory**

If you do not have DeepSeek-V4-Flash running yet, start with the recipe this is
built on, which is the exact configuration benchmarked here:

**github.com/tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark**

Get that serving tokens first. Then add the video layer.

Two Sparks. A 1M-context coding model and a video factory. At the same time.
