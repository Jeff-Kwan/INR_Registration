"""In-memory adapter to pinned Learn2Reg AbdomenCTCT public-validation metrics."""
import ast
import hashlib
from functools import lru_cache
import json
from pathlib import Path
import types
import numpy as np
import nibabel as nib
from scipy.ndimage import find_objects, map_coordinates
import surface_distance
import torch
from medical.registration.cuda_kernels import supported

PROTOCOL_DIR=Path(__file__).parent/'protocols'
REVISION='88475095e35442087a24439ff34ec09f54bc7dac'
PROTOCOLS={name:json.loads((PROTOCOL_DIR/f'{name}_val.json').read_text()) for name in ('abdomenctct','oasis')}
PROTOCOL=PROTOCOLS['abdomenctct']

class NumpyCompat:
    # The pinned utility uses the NumPy 1 spelling removed by NumPy 2.
    def __getattr__(self,name):
        return np.nan if name=='NAN' else getattr(np,name)

@lru_cache(maxsize=1)
def official_functions():
    provenance=json.loads((PROTOCOL_DIR/'provenance.json').read_text())
    for name,expected in provenance['files_sha256'].items():
        if hashlib.sha256((PROTOCOL_DIR/name).read_bytes()).hexdigest()!=expected:
            raise RuntimeError(f'Pinned official protocol source changed: {name}')
    source=ast.parse((PROTOCOL_DIR/'l2r_utils.py').read_text())
    names={'jacobian_determinant','compute_dice','compute_hd95'}
    nodes=[n for n in source.body if isinstance(n,ast.FunctionDef) and n.name in names]
    namespace={**vars(surface_distance),'np':NumpyCompat(),'scipy':__import__('scipy')}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'pinned_l2r_metrics','exec'),namespace)
    return types.SimpleNamespace(**{n:namespace[n] for n in names})

def label_path(volume):
    image=Path(volume['path'])
    if image.parent.name!='imagesTr': raise ValueError('Expected official imagesTr inputs')
    return image.parent.parent/'labelsTr'/image.name

def validate_case(case,protocol="abdomenctct"):
    specification=PROTOCOLS[protocol]
    for volume in case.values():
        label=nib.load(label_path(volume))
        if label.shape!=volume['data'].shape or not np.allclose(label.affine,volume['affine']):
            raise ValueError('Evaluation labels must share the released image grid')
    names={tuple(Path(p[k]).name for k in ('fixed','moving')) for p in specification['eval_pairs']}
    if (Path(case['fixed']['path']).name,Path(case['moving']['path']).name) not in names:
        raise ValueError(f'Pair is not in the official {protocol} validation list')

def dense_mapping(predict,shape,chunk=8192):
    result=np.empty((*shape,3),dtype=np.float32)
    flat=result.reshape(-1,3)
    for start in range(0,len(flat),chunk):
        indices=np.arange(start,min(start+chunk,len(flat)))
        points=np.column_stack(np.unravel_index(indices,shape)).astype(np.float32)
        flat[start:start+len(points)]=predict(points)
    return result

def json_metric(value):
    value=float(value)
    return value if np.isfinite(value) else None

@lru_cache(maxsize=1)
def _surface_distances():
    """Run the pinned surface-distance code with its tuple sort in NumPy."""
    original = surface_distance.compute_surface_distances
    def sort_surfels(distances, areas):
        # Python sorted(zip(distances, areas)) orders ties by surface area.
        order = np.lexsort((areas, distances))
        return distances[order], areas[order]
    namespace = {**original.__globals__, '_sort_distances_surfels': sort_surfels}
    return types.FunctionType(original.__code__, namespace)


def _label_metrics(fixed, moving, warped, labels, official):
    # Released labels are integers. Preserve the original equality semantics
    # for user-supplied label images with non-integer dtypes.
    if any(array.dtype.kind not in 'biu' for array in (fixed, moving, warped)):
        return (official.compute_dice(fixed, moving, warped, labels),
                official.compute_hd95(fixed, moving, warped, labels))
    maximum = max(labels)
    boxes = [find_objects(array, max_label=maximum) for array in (fixed, moving, warped)]
    dice, hd95 = [], []
    for label in labels:
        truth, original, prediction = (box[label - 1] for box in boxes)
        if truth is None or original is None:
            dice.append(np.nan)
            hd95.append(np.nan)
            continue
        # All foreground voxels for this comparison are inside this box.
        roi = truth if prediction is None else tuple(
            slice(min(a.start, b.start), max(a.stop, b.stop)) for a, b in zip(truth, prediction))
        mask, warped_mask = fixed[roi] == label, warped[roi] == label
        dice.append(surface_distance.compute_dice_coefficient(mask, warped_mask))
        distances = _surface_distances()(mask, warped_mask, np.ones(3))
        hd95.append(surface_distance.compute_robust_hausdorff(distances, 95.0))
    return (np.nanmean(dice), dice), (np.nanmean(hd95), hd95)


def _interior_jacobian(mapping):
    """Pinned central differences on the two-voxel interior, in FP64 slabs."""
    shape = mapping.shape[:3]
    interior = tuple(max(0, size - 4) for size in shape)
    tensor = isinstance(mapping, torch.Tensor)
    if tensor:
        determinant = mapping.new_empty(interior, dtype=torch.float64)
    else:
        determinant = np.empty(interior, dtype=np.float64)
    depth = 32 if tensor else 8
    for start in range(2, shape[0] - 2, depth):
        stop = min(start + depth, shape[0] - 2)
        slab = mapping[start - 1:stop + 1, 1:-1, 1:-1]
        origins = (start - 1, 1, 1)
        rows = []
        for axis in range(3):
            origin, count = origins[axis], slab.shape[axis]
            coord_shape = [1, 1, 1]
            coord_shape[axis] = count
            if tensor:
                coord = torch.arange(origin, origin + count, device=slab.device)
                component = slab[..., axis].double()
            else:
                coord = np.arange(origin, origin + count)
                component = slab[..., axis].astype(np.float64, copy=False)
            disp = component - coord.reshape(coord_shape)
            gradients = [(disp[2:, 1:-1, 1:-1] - disp[:-2, 1:-1, 1:-1]) * 0.5,
                         (disp[1:-1, 2:, 1:-1] - disp[1:-1, :-2, 1:-1]) * 0.5,
                         (disp[1:-1, 1:-1, 2:] - disp[1:-1, 1:-1, :-2]) * 0.5]
            gradients[axis] += 1.0
            rows.append(gradients)
        # Match both matrix orientation and operation order in l2r_utils.py.
        a, b, c = rows
        determinant[start - 2:stop - 2] = (a[0] * (b[1] * c[2] - c[1] * b[2])
            - a[1] * (b[0] * c[2] - c[0] * b[2])
            + a[2] * (b[0] * c[1] - c[0] * b[1]))
    return determinant


def evaluate(case,prediction,protocol="abdomenctct"):
    """Prediction: absolute moving XYZ voxels on fixed XYZ grid, or point callable."""
    fixed=np.asarray(nib.load(label_path(case['fixed'])).dataobj)
    moving=np.asarray(nib.load(label_path(case['moving'])).dataobj)
    labels=next(method['labels'] for method in PROTOCOLS[protocol]['evaluation_methods'] if method['metric']=='dice')
    result = evaluate_labels(fixed, moving, prediction, labels)
    result.update(dataset=protocol, evaluator_revision=REVISION)
    return result


def evaluate_labels(fixed, moving, prediction, labels):
    """Learn2Reg metrics for arrays on their preserved evaluation grid."""
    functions=official_functions()
    # The Triton EDT tiles a full axis of the released native grids. Larger
    # and user-supplied label formats retain the generic CPU implementation.
    cuda_native = (isinstance(prediction, torch.Tensor) and supported(prediction.device)
                   and fixed.dtype == moving.dtype == np.uint8 and max(fixed.shape) <= 256)
    if cuda_native:
        if prediction.shape != (*fixed.shape, 3) or not torch.isfinite(prediction).all():
            raise ValueError('Expected a finite fixed-to-moving XYZ voxel map')
        from medical.learn2reg_cuda import label_metrics
        (dice, dices), (hd95, distances) = label_metrics(fixed, moving, prediction, labels)
        jac = _interior_jacobian(prediction.detach())
        shifted = (jac + 3).clamp_(1e-9, 1e9)
        sdlog = shifted.log_().std(correction=0).item()
        # Clipping to 1e-9 guarantees the official shifted count is zero.
        shifted_foldings = 0.0
        # An integer count divided once matches NumPy's boolean mean.
        # CUDA mean can multiply by a rounded reciprocal of the voxel count.
        folding = torch.count_nonzero(jac <= 0).item() / jac.numel() if jac.numel() else np.nan
    else:
        if isinstance(prediction, torch.Tensor):
            prediction = prediction.detach().cpu().numpy()
        mapping=dense_mapping(prediction,fixed.shape) if callable(prediction) else np.asarray(prediction)
        if mapping.shape!=(*fixed.shape,3) or not np.isfinite(mapping).all():
            raise ValueError('Expected a finite fixed-to-moving XYZ voxel map')
        warped=map_coordinates(moving,mapping.transpose(3,0,1,2),order=0)
        (dice,dices),(hd95,distances)=_label_metrics(fixed,moving,warped,labels,functions)
        jac=_interior_jacobian(mapping)
        shifted = (jac + 3).clip(1e-9,1e9)
        sdlog, shifted_foldings, folding = np.log(shifted).std(), (shifted <= 0).sum(), (jac <= 0).mean()
    return {'DSC':{'mean':json_metric(dice),'detailed':[json_metric(v) for v in dices]},
            'HD95':{'mean':json_metric(hd95),'detailed':[json_metric(v) for v in distances],'units':'voxels (official unit spacing)'},
            'LogJacDetStd':{'mean':float(sdlog),'definition':'std(log(clip(detJ+3,1e-9,1e9)))'},
            'num_foldings':{'mean':float(shifted_foldings),'definition':'official shifted/clipped count'},
            'folding_fraction':float(folding),'folding_status':'evaluated_unshifted_interior_detJ',
            'labels':labels,'missing_or_nonfinite_metric_policy':'null; never silently replaced with zero'}
