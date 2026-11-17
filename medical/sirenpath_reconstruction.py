"""Whole-grid PSNR and valid-window Patch NCC for the endpoint reconstructions."""
import math
import torch
import torch.nn.functional as F
from medical.models.siren import coordinates_from_indices


def _box_mean(image, window):
    result = image[None, None]
    for axis in range(3):
        kernel = tuple(window if k == axis else 1 for k in range(3))
        result = F.avg_pool3d(result, kernel, stride=1)
    return result[0, 0]


def patch_ncc(prediction, truth, window=9, epsilon=1e-8, slab=16):
    """Signed Pearson NCC at every valid 9³ window; constant truth is excluded."""
    if prediction.shape != truth.shape or min(truth.shape) < window:
        raise ValueError('Matching volumes must contain a full NCC window')
    total = truth.new_zeros((), dtype=torch.float64)
    count = 0
    for start in range(0, truth.shape[0]-window+1, slab):
        stop = min(start+slab, truth.shape[0]-window+1)+window-1
        a, b = prediction[start:stop], truth[start:stop]
        # FP64 moments avoid cancellation in low-variance background windows.
        a, b = a.double(), b.double()
        ma, mb = _box_mean(a, window), _box_mean(b, window)
        va = (_box_mean(a.square(), window)-ma.square()).clamp_min(0)
        vb = (_box_mean(b.square(), window)-mb.square()).clamp_min(0)
        covariance = _box_mean(a*b, window)-ma*mb
        valid = vb > epsilon
        corr = (covariance/(va*vb).sqrt().clamp_min(epsilon)).clamp(-1, 1)
        total += corr[valid].sum()
        count += int(valid.sum())
    return dict(mean=float(total/count) if count else None, scored_windows=count)


@torch.no_grad()
def reconstruct_endpoint(model, image, chunk=65536):
    predicted = torch.empty_like(image)
    for start in range(0, image.numel(), chunk):
        indices = torch.arange(start, min(start+chunk, image.numel()), device=image.device)
        coords = coordinates_from_indices(indices, image.shape).to(image.dtype)
        predicted.flatten()[indices] = model(coords).flatten()
    return predicted


@torch.no_grad()
def endpoint_scores(pair, images, *, chunk=65536, raw=False, predictions=None):
    result = dict(scope='whole prepared grid; independent minmax inputs', data_range=1.,
                  patch_window=[9,9,9], patch_stride=1, patch_boundary='valid windows only',
                  patch_constant_truth='variance <=1e-8 excluded; no anatomical masks',
                  patch_ncc='signed Pearson; denominator max(sqrt(var_pred*var_truth),1e-8)')
    if raw and predictions is not None:
        raise ValueError('Raw identity and model reconstructions are different controls')
    for index, (name, image) in enumerate(zip(('moving', 'fixed'), images)):
        if raw:
            predicted = image
        elif predictions is not None:
            predicted = predictions[index]
            if predicted.shape != image.shape:
                raise ValueError('Reconstruction must share the original endpoint grid')
        else:
            model = pair.source if name == 'moving' else pair.target
            predicted = reconstruct_endpoint(model, image, chunk)
        mse = float((predicted.double()-image.double()).square().mean())
        if not math.isfinite(mse):
            raise RuntimeError('Nonfinite endpoint reconstruction')
        result[name] = dict(mse=mse, psnr_db=-10*math.log10(mse) if mse else None,
                            perfect_reconstruction=mse == 0, psnr_positive_infinity=mse == 0,
                            patch_ncc=patch_ncc(predicted, image))
    finite = [result[k]['psnr_db'] for k in ('moving', 'fixed')]
    result['PSNR'] = dict(mean=sum(finite)/2 if all(v is not None for v in finite) else None,
                          positive_infinity=any(result[k]['perfect_reconstruction'] for k in ('moving','fixed')))
    values = [result[k]['patch_ncc']['mean'] for k in ('moving','fixed')]
    result['PatchNCC'] = dict(mean=sum(values)/2 if all(v is not None for v in values) else None)
    return result
