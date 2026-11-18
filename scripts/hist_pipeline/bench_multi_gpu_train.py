#!/usr/bin/env python3
# scripts/hist_pipeline/bench_multi_gpu_train.py
"""
How much would training the two-look network on all GPUs save? Times the network part of a training
step (forward of both stages, loss, backward, gradient clip, optimizer step) for:

  one GPU at batch sizes 32 to 1024 (train_selfcal_online.py uses 128 now), and
  all GPUs at once with the batch split between them, gradients summed onto GPU 0 and the weights sent
  back (the extra work data-parallel training adds)

Histogram building is not included: it is already split over the GPUs. Inputs are random histograms of
the real shapes; the timing doesn't depend on their values.
  python3 -m scripts.hist_pipeline.bench_multi_gpu_train
"""
import threading
import time

import torch
import torch.cuda.comm as comm

from scripts.scripts.softdwell_pipeline.model import HardSelfCalibratingFusionNet
from scripts.softdwell_pipeline.metrics import LogCoshLoss

K, J, OUT = 24, 12, 9


def make(dev):
    torch.manual_seed(0)
    m = HardSelfCalibratingFusionNet(K, J, K, J, num_outputs=OUT, backbone="convnext_v2").to(dev)
    return m, torch.optim.AdamW(m.parameters(), lr=3e-3)


def inputs(b, dev):
    g = torch.Generator(device=dev).manual_seed(1)
    return (torch.rand(b, K, J, J, device=dev, generator=g) * 4, torch.rand(b, K, J, J, device=dev, generator=g) * 4,
            torch.randn(b, OUT, device=dev, generator=g), torch.rand(b, 2, device=dev, generator=g) * 5)


def step(m, opt, x, do_opt=True):
    fixed, selfcal, y, lv = x
    crit = LogCoshLoss()
    opt.zero_grad(set_to_none=True)
    f, base, amp = m.fixed_stage(fixed)
    p = m.finish(f, selfcal, base, amp)
    loss = crit(p, y) + 0.3 * crit(torch.stack([base, amp], 1), lv)
    loss.backward()
    if do_opt:
        torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
        opt.step()


def timed(fn, reps=60, warm=10):
    for _ in range(warm):
        fn()
    for d in range(torch.cuda.device_count()):
        torch.cuda.synchronize(d)
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    for d in range(torch.cuda.device_count()):
        torch.cuda.synchronize(d)
    return 1000 * (time.perf_counter() - t0) / reps


def main():
    n = torch.cuda.device_count()
    devs = [torch.device(f"cuda:{i}") for i in range(n)]
    print(f"{n} x {torch.cuda.get_device_name(0)}")
    m0, o0 = make(devs[0])
    print(f"parameters: {sum(p.numel() for p in m0.parameters()):,}")
    one = {}
    for bs in (32, 64, 128, 256, 512, 1024):
        xb = inputs(bs, devs[0])
        one[bs] = timed(lambda: step(m0, o0, xb))
        print(f"one GPU, batch {bs:5d}: {one[bs]:6.1f} ms per step | {one[bs] * 19968 / bs / 1000:5.2f} s of network "
              f"time per epoch of 19,968 samples")
    models = [make(d) for d in devs]
    for bs in (128, 512, 1024):
        xs = [inputs(bs // n, d) for d in devs]
        c = timed(lambda: dp_step(models, xs), reps=30)
        print(f"{n} GPUs, total batch {bs:5d} ({bs // n} each), with gradient exchange: {c:6.1f} ms per step "
              f"(one GPU: {one[bs]:6.1f} ms)")
    print("note: the multi-GPU case uses Python threads and a simple per-tensor exchange; real "
          "DistributedDataParallel overlaps the exchange with the backward pass, so it would usually be faster")


def dp_step(models, xs):
    devs = [next(m.parameters()).device for m, _ in models]
    n = len(models)
    if True:
        def work(i):
            with torch.cuda.device(devs[i]):
                step(models[i][0], models[i][1], xs[i], do_opt=False)
        ts = [threading.Thread(target=work, args=(i,)) for i in range(n)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        params = [list(m.parameters()) for m, _ in models]
        for j in range(len(params[0])):                         # sum the gradients onto GPU 0
            g = comm.reduce_add([p[j].grad for p in params], destination=0)
            params[0][j].grad = g / n
        torch.nn.utils.clip_grad_norm_(params[0], 5.0)
        models[0][1].step()
        with torch.no_grad():                                   # send the new weights back
            copies = comm.broadcast_coalesced([p.data for p in params[0]], [d.index for d in devs])
            for i in range(1, n):
                for p, c in zip(params[i], copies[i]):
                    p.copy_(c)



if __name__ == "__main__":
    main()
