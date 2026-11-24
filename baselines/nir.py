"""NIR MRI recipe through the released loop, with stable IEEE FP32 NCC."""
from tempfile import TemporaryDirectory
import time
import math

import numpy as np
import torch
from torch.utils.data import DataLoader

from .common import cpu_state, execution, finish, preflight
from .sources import native_modules, upstream_imports
from .precision import ieee_fp32


class ArrayDataset:
    """In-memory equivalent of native Single_CT_Brain's Identity data loader."""
    def __init__(self, array):
        self.v = array
        self.shape = array.shape
        self.maxlen = None
        self.channels = 1

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return self.v[..., np.newaxis]


def prepare_array(array, preprocessing):
    if preprocessing == 'preserve_prepared':
        return array
    if preprocessing == 'prepared_0_1':
        if array.min() < 0 or array.max() > 1:
            raise ValueError('NIR prepared_0_1 inputs must already lie in [0,1]')
        return array
    if preprocessing == 'independent_minmax_0_1':
        span = float(array.max() - array.min())
        if span <= 0:
            raise ValueError('Cannot min/max normalize a constant NIR input')
        return ((array - array.min()) / span).astype(np.float32)
    raise ValueError(f'Unknown NIR preprocessing: {preprocessing}')


def native_output_voxels(normalized, shape, maxlen):
    """Voxel conversion in the released util.warp_3d_volume inference helper."""
    return np.asarray(normalized)/2*np.asarray(maxlen)+(np.asarray(shape)-1)/2


def build_model(native, modules, efficient, deform, moving):
    # Exactly the released CLI model construction for these explicit variants.
    if native['model_type'] == 'gaussian_sine':
        field = efficient.EfficientMLP(in_features=3, out_features=3, num_hidden_layers=native['num_hidden_layers'],
                                       mode='gaussian', gaussian_scale=native['gaussian_scale'], nonlinearity='sine')
        return deform.DeformedField(field, 'residual', moving)
    if native['model_type'] == 'ode_gaussian_sine':
        encoding = efficient.PosEncodingFourier(in_features=3, method='gaussian',
                                               gaussian_scale=native['gaussian_scale'], embedding_size=128)
        dynamic = modules.FCBlock(in_features=256, out_features=3, num_hidden_layers=native['num_hidden_layers'],
                                  hidden_features=256, outermost_linear=True, nonlinearity='sine')
        return deform.DiffeoDeformedField(deform.ODEFunc(encoding, dynamic), moving)
    raise ValueError('Connected NIR variants: gaussian_sine or ode_gaussian_sine')


def run_case(case, config, device):
    preflight(case, config, 'nir')
    native = config['native']
    losses = native['loss_type'].split('+')
    preprocessing = config['preprocessing']
    arrays = {key: prepare_array(case[key]['data'], preprocessing) for key in ('fixed', 'moving')}
    started = time.monotonic()
    with upstream_imports('nir'), execution(device, config.get('seed', 42), cuda_required=True), ieee_fp32():
        dataio, modules, efficient, deform, training = native_modules('nir')
        dataset = ArrayDataset(arrays['fixed'])
        sampled = dataio.Implicit3DWrapper(dataset, sample_method=native['sample_mtd'], sample_exp=native['sample_exp'])
        dataset.maxlen = sampled.maxlen
        loader = DataLoader(sampled, shuffle=True, batch_size=native['batch_size'], pin_memory=True, num_workers=0)
        # Preserve the source loop's complete DataLoader passes. Its requested
        # iteration limit can be exceeded by the final pass; record both counts.
        requested_steps = native['num_epochs']
        native_steps = math.ceil(requested_steps / len(loader)) * len(loader)
        moving = torch.from_numpy(arrays['moving']).float().cuda()
        fixed = torch.from_numpy(arrays['fixed']).float().cuda()
        moving_model = dataio.Single_CT_Brain_Model(moving, dataset.shape, sampled.maxlen)
        fixed_model = dataio.Single_CT_Brain_Model(fixed, dataset.shape, sampled.maxlen)
        model = build_model(native, modules, efficient, deform, moving).cuda().train()
        # The source creates a writer without closing it. Close it on success and
        # exceptions so all native logs/checkpoints live only in temporary scratch.
        writers = []
        writer_type = training.SummaryWriter
        def writer_factory(*args, **kwargs):
            writer = writer_type(*args, **kwargs)
            writers.append(writer)
            return writer
        with TemporaryDirectory(prefix='nir-') as scratch:
            try:
                training.SummaryWriter = writer_factory
                training.train_efficient_pairwise(
                    model, moving_model, fixed_model, train_dataloader=loader,
                    epochs=native['num_epochs'], lr=native['lr'],
                    steps_til_summary=native['steps_til_summary']//native['batch_size'], epochs_til_checkpoint=native['epochs_til_ckpt'],
                    model_dir=scratch + '/native', summary_fn=None, loss_types=losses,
                    loss_weight=native['loss_weight'], maxlen=sampled.maxlen)
            finally:
                training.SummaryWriter = writer_type
                for writer in writers:
                    writer.close()
            model.eval().requires_grad_(False)
            def predict(points):
                coordinates = dataio.normalize_coords(points.copy(), dataset.shape, sampled.maxlen).reshape(1, -1, 3).cuda()
                with torch.no_grad():
                    normalized = model.deform(coordinates)[0].cpu()
                # Match util.warp_3d_volume used by the released test script.
                # Its scale is maxlen, unlike dataio.unnormalize_coords (maxlen-1).
                return native_output_voxels(normalized.numpy(), dataset.shape, sampled.maxlen)
            predicted=predict
            checkpoint = {'kind': 'network_weights', 'state': cpu_state(model)}
        return finish('nir', case, config, predicted, checkpoint, started, variant=native['model_type'],
                      sampler=native['sample_mtd'], preprocessing=preprocessing,
                      output_coordinates='native util.warp_3d_volume maxlen scale',
                      optimization_roi='whole_image', protocol='NIR-D MRI LNCC recipe; paper spectral settings; released native loop with IEEE FP32 localization',
                      precision='FP32; autocast/GradScaler/TF32 convolution and matmul disabled',
                      gaussian_encoding='128 Gaussian vectors, sigma3; cos/sin(2*pi*p*omega)',
                      sine_frequency=30,
                      requested_steps=requested_steps, native_loop_steps=native_steps,
                      learning_rate_schedule='native linear decay to 0.1*lr over 2000 steps, updated per loader pass')
