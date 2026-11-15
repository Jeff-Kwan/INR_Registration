"""Opt-in synthetic SIREN profiles. No data, optimizer or checkpoint recovery."""
import argparse
import gc
import json
from pathlib import Path
import statistics
import time

import torch

from medical.models.siren import PairedSiren
from medical.registration.objective import TrainingBatch, endpoint_loss
from medical.registration.recovery import derivative_grid
from medical.registration.symmetric_registration import SymmetricPairRegistration


def measure(function, pair, *, backward, warmup, repeats, device):
    def run():
        pair.zero_grad(set_to_none=True)
        value = function()
        if backward:
            value.backward()
        del value
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    run()
    torch.cuda.synchronize(device)
    cold_ms = (time.perf_counter() - start) * 1000
    for _ in range(warmup):
        run()
    torch.cuda.synchronize(device)
    pair.zero_grad(set_to_none=True)
    gc.collect()
    baseline = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    durations = []
    for _ in range(repeats):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        run()
        torch.cuda.synchronize(device)
        durations.append((time.perf_counter() - start) * 1000)
    return dict(cold_ms=cold_ms, median_ms=statistics.median(durations),
                min_ms=min(durations),
                peak_mib=torch.cuda.max_memory_allocated(device) / 2**20,
                incremental_peak_mib=(torch.cuda.max_memory_allocated(device)-baseline) / 2**20)


def profile(args):
    device = torch.device(args.device)
    if device.type != 'cuda':
        raise ValueError('This profiler requires an explicitly selected CUDA device')
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    torch.manual_seed(42)
    pair = PairedSiren(width=args.width, layers=args.layers, omega=args.omega).to(device)
    with torch.no_grad():
        for parameter in pair.target.parameters():
            parameter.add_(torch.randn_like(parameter) * .0001)
    shape = (65, 61, 57)
    coordinates = torch.rand(args.batch, 3, device=device) * 2 - 1
    batch = TrainingBatch(coordinates, torch.rand(args.batch, device=device),
                          torch.rand(args.batch, device=device))
    image = torch.rand(shape, device=device)
    branch = SymmetricPairRegistration(image, image.flip(0), torch.eye(4), grid_size=8,
        pairs_per_cell=2, particles=4096, chunk=args.derivative_chunk, recompute=args.recompute,
        gaussian_kernel='three_tap', crop_fraction=.5, random_timesteps=True)
    timepoint = image.new_tensor(.37)
    basic_coordinates = torch.rand(args.dense_points, 3, device=device) * 2 - 1
    derivative_coordinates = coordinates[:2048]
    report = dict(torch=torch.__version__, cuda=torch.version.cuda, device=str(device),
        gpu=torch.cuda.get_device_name(device), dtype='float32', seed=42,
        matmul_precision=torch.get_float32_matmul_precision(),
        tf32=torch.backends.cuda.matmul.allow_tf32, settings=vars(args).copy(), modes={})
    for mode in args.modes:
        compiled = mode != 'eager'
        branch.compile_derivatives = compiled
        branch.compile_velocity = mode == 'fused'
        def endpoint(endpoint_mode='mse'):
            return endpoint_loss(pair, batch, {'endpoint':endpoint_mode},
                                 compile_endpoints=compiled)['total']
        def differential():
            outputs = pair.derivatives(derivative_coordinates, timepoint, shape,
                        differential=True, compile_derivatives=compiled)
            return sum(value.square().mean() for value in outputs)
        def combined():
            terms = branch.loss(pair)
            return endpoint() + .1 * terms['grid_pdlk'] + .01 * terms['registration']
        def basic():
            with torch.no_grad():
                return pair.derivatives(basic_coordinates, timepoint, shape,
                                        compile_derivatives=compiled)
        def dense_grid():
            return derivative_grid(pair, (128, 128, 64), .37, chunk=args.inference_chunk,
                                   compile_derivatives=compiled)
        stages = dict(endpoint=(endpoint, True), differential=(differential, True),
                      combined=(combined, True), basic=(basic, False),
                      endpoint_ncc=(lambda: endpoint('ncc'), True), dense_grid=(dense_grid, False))
        timings = {}
        for name in args.stages:
            function, backward = stages[name]
            timings[name] = measure(function, pair, backward=backward,
                warmup=args.warmup, repeats=args.repeats, device=device)
            print(json.dumps(dict(mode=mode, stage=name, **timings[name])), flush=True)
        report['modes'][mode] = timings
        if args.trace and mode == args.modes[-1]:
            pair.zero_grad(set_to_none=True)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as trace:
                combined().backward()
                torch.cuda.synchronize(device)
            trace.export_chrome_trace(args.trace)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', help='Execute the synthetic GPU profiles')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--modes', nargs='+', choices=['eager', 'compiled', 'fused'],
                        default=['eager', 'compiled', 'fused'])
    parser.add_argument('--stages', nargs='+', choices=['endpoint', 'endpoint_ncc', 'differential', 'combined', 'basic', 'dense_grid'],
                        default=['endpoint', 'differential', 'combined', 'basic', 'endpoint_ncc'])
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--layers', type=int, default=4)
    parser.add_argument('--omega', type=float, default=30.)
    parser.add_argument('--batch', type=int, default=32768)
    parser.add_argument('--dense-points', type=int, default=262144)
    parser.add_argument('--derivative-chunk', type=int, default=4096)
    parser.add_argument('--inference-chunk', type=int, default=65536)
    parser.add_argument('--recompute', action='store_true')
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--output', help='Optional fresh JSON file under an ignored results directory')
    parser.add_argument('--trace', help='Optional fresh profiler trace under an ignored results directory')
    args = parser.parse_args()
    if min(args.width, args.layers, args.batch, args.dense_points, args.derivative_chunk,
           args.inference_chunk, args.repeats) < 1 or args.warmup < 0:
        parser.error('Sizes/repeats must be positive and warmup nonnegative')
    if args.batch < 2048:
        parser.error('Batch must contain the 2048 default derivative sites')
    for path in (args.output, args.trace):
        if path and Path(path).exists():
            parser.error(f'Refusing to overwrite {path}')
    if not args.run:
        print(json.dumps(dict(plan=vars(args), note='Pass --run to execute; no GPU work performed'), indent=2))
        return
    for path in (args.output, args.trace):
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
    report = profile(args)
    if args.output:
        with Path(args.output).open('x') as output:
            json.dump(report, output, indent=2)


if __name__ == '__main__':
    main()
