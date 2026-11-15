"""In-memory physical working grid; NIfTI geometry is XYZ, tensors are ZYX."""
import math
import numpy as np
from scipy.ndimage import affine_transform, gaussian_filter
import torch
from medical.registration.geometry import identity_grid, sample


def geometry(shape, affine):
    if len(shape) != 3 or any(type(n) is not int or n < 2 for n in shape):
        raise ValueError('Reference shape must contain three integer sizes >= 2')
    a = np.asarray(affine, dtype=np.float64)
    if a.shape != (4, 4) or not np.isfinite(a).all() or not np.allclose(a[3], [0,0,0,1]):
        raise ValueError('Expected finite homogeneous affine')
    spacing = np.linalg.norm(a[:3,:3], axis=0)
    if (spacing <= 0).any() or not np.allclose(a[:3,:3].T@a[:3,:3], np.diag(spacing**2), rtol=1e-6, atol=1e-8):
        raise ValueError('Working grid requires orthogonal nondegenerate axes')
    return {'shape':list(shape), 'affine':a.tolist()}


def validate_options(options):
    if not isinstance(options,dict) or set(options)-{'spacing_mm','reference_geometry','crop_extent_mm','recovery_spacing_mm'} or options.get('spacing_mm') != 1.0:
        raise ValueError('working_grid requires spacing_mm=1.0 and optional reference_geometry')
    recovery_spacing=options.get('recovery_spacing_mm','original')
    if recovery_spacing != 'original' and (type(recovery_spacing) not in (int,float) or not math.isfinite(recovery_spacing) or recovery_spacing < 1):
        raise ValueError("recovery_spacing_mm must be original or finite and >= 1 mm")
    extent=options.get('crop_extent_mm')
    if extent is not None and (type(extent) is not int or extent < 1):
        raise ValueError('crop_extent_mm must be a positive integer at 1 mm spacing')
    if 'reference_geometry' in options:
        ref=options['reference_geometry']
        if not isinstance(ref,dict) or set(ref)!={'shape','affine'}:
            raise ValueError('reference_geometry requires shape and affine')
        geometry(ref['shape'],ref['affine'])


def canonical_geometry(reference):
    ref=geometry(reference['shape'],reference['affine'])
    a=np.asarray(ref['affine']); spacing=np.linalg.norm(a[:3,:3],axis=0)
    length=(np.asarray(ref['shape'])-1)*spacing
    rounded=np.rint(length)
    length=np.where(np.abs(length-rounded)<=1e-6,rounded,length)
    shape=np.ceil(length).astype(int)+1
    out=np.eye(4);out[:3,:3]=a[:3,:3]/spacing
    center=a[:3,:3]@((np.asarray(ref['shape'])-1)/2)+a[:3,3]
    out[:3,3]=center-out[:3,:3]@((shape-1)/2)
    return geometry(shape.tolist(),out)


def recovery_geometry(working, spacing_mm=1.):
    """Same voxel-center physical bounds; spacing is a maximum requested step."""
    spacing = np.asarray(spacing_mm, dtype=np.float64)
    if spacing.shape not in ((), (3,)) or not np.isfinite(spacing).all() or (spacing <= 0).any():
        raise ValueError('Recovery spacing must be a positive scalar or XYZ triple')
    if (spacing == 1).all():
        return geometry(working['shape'],working['affine'])
    a=np.asarray(working['affine'],dtype=np.float64).copy()
    extent=np.asarray(working['shape'])-1
    intervals=np.maximum(1,np.ceil(extent/spacing).astype(int))
    a[:3,:3]=a[:3,:3]*(extent/intervals)
    return geometry((intervals+1).tolist(),a)


def requested_recovery_geometry(working, reference, spacing):
    # Frozen original reference spacing also preserves comparisons of resampled inputs.
    if spacing == 'original':
        spacing = np.linalg.norm(np.asarray(reference['affine'])[:3, :3], axis=0)
    return recovery_geometry(working, spacing)


def resample(data, affine, target, *, mask=False):
    source=geometry(list(data.shape),affine)
    same=source['shape']==target['shape'] and np.allclose(source['affine'],target['affine'],rtol=0,atol=1e-6)
    spacing=np.linalg.norm(np.asarray(affine)[:3,:3],axis=0)
    sigma=np.zeros(3) if mask else .5*np.sqrt(np.maximum(spacing**-2-1,0))
    if same:
        return data, {'interpolation':'bypass','antialias_sigma_source_voxels':[0.,0.,0.]}
    values=np.asarray(data,dtype=np.float32)
    if np.any(sigma):values=gaussian_filter(values,sigma,mode='nearest')
    transform=np.linalg.solve(np.asarray(affine),np.asarray(target['affine']))
    values=affine_transform(values,transform[:3,:3],offset=transform[:3,3],output_shape=target['shape'],
                            order=0 if mask else 1,mode='constant' if mask else 'nearest',cval=0,prefilter=False)
    return values, {'interpolation':'nearest' if mask else 'trilinear','boundary':'zero' if mask else 'nearest',
                    'antialias_sigma_source_voxels':sigma.tolist()}


def prepare(case, options, device, masks=None):
    validate_options(options)
    for key in ('moving','fixed'):geometry(list(case[key]['data'].shape),case[key]['affine'])
    if (case['moving']['data'].shape != case['fixed']['data'].shape or
            not np.allclose(case['moving']['affine'],case['fixed']['affine'],rtol=1e-5,atol=1e-5)):
        raise ValueError('Prepared endpoints must share shape and affine')
    reference=options.get('reference_geometry',geometry(list(case['fixed']['data'].shape),case['fixed']['affine']))
    reference=geometry(reference['shape'],reference['affine']);working=canonical_geometry(reference)
    metadata={'recovery_spacing_mm':options.get('recovery_spacing_mm','original'),
              'recovery_geometry':requested_recovery_geometry(working,reference,options.get('recovery_spacing_mm','original')), 'version':1,'reference_geometry':reference,'working_geometry':working,'spacing_mm':1.,'endpoints':{}}
    arrays=[];out_masks=[]
    for index,key in enumerate(('moving','fixed')):
        volume=case[key];values,details=resample(volume['data'],volume['affine'],working)
        lo,hi=float(values.min()),float(values.max())
        if not math.isfinite(lo+hi) or hi<=lo:raise ValueError('Cannot normalize constant or nonfinite working image')
        arrays.append(torch.from_numpy(((values-lo)/(hi-lo)).transpose(2,1,0).copy()).to(device))
        details.update(normalization={'min':lo,'max':hi},original_geometry=geometry(list(volume['data'].shape),volume['affine']))
        if masks is not None:
            m,mask_details=resample(masks[index],volume['affine'],working,mask=True)
            if not m.any():raise ValueError('Resampled endpoint mask is empty')
            out_masks.append(torch.from_numpy(m.transpose(2,1,0).copy()).to(device).bool())
            details['mask']=mask_details
        metadata['endpoints'][key]=details
    a=np.asarray(working['affine']).copy();a[:3,:3]=a[:3,:3][:,::-1]
    return arrays,a,out_masks,metadata


def transform_points(points, affine):
    return points@affine[:3,:3].T+affine[:3,3]


def predictor(mapping, metadata, query_affine, output_affine):
    """Original query XYZ -> original output XYZ, interpolating displacement."""
    mapping_grid=metadata.get('recovery_geometry',metadata['working_geometry'])
    if list(reversed(mapping_grid['shape']))!=list(mapping.shape[:3]):
        raise ValueError('Map shape differs from recorded recovery geometry')
    working=np.asarray(mapping_grid['affine'])
    to_work=np.linalg.solve(working,np.asarray(query_affine))
    from_work=np.linalg.solve(np.asarray(output_affine),working)
    displacement=mapping-identity_grid(mapping.shape[:3],mapping.device)
    def predict(points):
        query=transform_points(np.asarray(points,dtype=np.float64),to_work)
        coords=torch.as_tensor(query,dtype=mapping.dtype,device=mapping.device).reshape(1,1,-1,3)
        delta=sample(displacement,coords).reshape(-1,3).cpu().numpy()
        return transform_points(query+delta,from_work).astype(np.float32)
    return predict


@torch.no_grad()
def original_reconstruction(pair, case, metadata, chunk):
    scores={'scope':'original_grid','data_range':1.0,'normalization':'working_grid_bounds_no_clipping'}
    working=metadata['working_geometry'];device=next(pair.parameters()).device
    for key,model in zip(('moving','fixed'),(pair.source,pair.target)):
        volume=case[key];shape=volume['data'].shape;bounds=metadata['endpoints'][key]['normalization']
        affine=np.linalg.solve(np.asarray(working['affine']),volume['affine']);error=0.
        for start in range(0,volume['data'].size,chunk):
            indices=np.arange(start,min(start+chunk,volume['data'].size))
            points=np.column_stack(np.unravel_index(indices,shape))
            points=transform_points(points,affine)
            coords=2*points/(np.asarray(working['shape'])-1)-1
            predicted=model(torch.as_tensor(coords,dtype=next(pair.parameters()).dtype,device=device)).flatten()
            truth=(volume['data'].reshape(-1)[indices]-bounds['min'])/(bounds['max']-bounds['min'])
            delta=predicted-torch.as_tensor(truth,device=device)
            error+=float(delta.square().sum(dtype=torch.float64))
        mse=error/volume['data'].size
        if not math.isfinite(mse):raise RuntimeError('Nonfinite original-grid reconstruction')
        scores[key]={'mse':mse,'psnr_db':-10*math.log10(mse) if mse else None,'perfect_reconstruction':mse==0}
    return scores


def validate_metadata(metadata, shape):
    if not isinstance(metadata,dict) or metadata.get('version')!=1:
        raise ValueError('Experimental recovery requires working-grid metadata')
    try:
        expected=canonical_geometry(metadata['reference_geometry'])
        working=metadata['working_geometry']
        if expected['shape']!=working['shape'] or not np.allclose(expected['affine'],working['affine'],rtol=0,atol=1e-8):
            raise ValueError('Inconsistent working-grid metadata')
        if 'recovery_geometry' in metadata:
            expected_recovery=requested_recovery_geometry(working,metadata['reference_geometry'],metadata.get('recovery_spacing_mm',1.))
            actual=metadata['recovery_geometry']
            if expected_recovery['shape']!=actual['shape'] or not np.allclose(expected_recovery['affine'],actual['affine'],rtol=0,atol=1e-8):
                raise ValueError('Inconsistent recovery geometry')
        if list(reversed(working['shape']))!=list(shape):raise ValueError('Recovery shape differs from working grid')
        for key in ('moving','fixed'):
            bounds=metadata['endpoints'][key]['normalization']
            if not np.isfinite([bounds['min'],bounds['max']]).all() or bounds['max']<=bounds['min']:raise ValueError('Invalid normalization')
    except (KeyError,TypeError) as exc:raise ValueError('Incomplete working-grid metadata') from exc


def restore_checkpoint(payload, device='cpu'):
    """Restore a case-runner experimental checkpoint, including its geometry."""
    from medical.models.siren import PairedSiren
    if 'working_grid' not in payload.get('config',{}):raise ValueError('Expected experimental working-grid checkpoint')
    metadata=payload.get('working_grid')
    if not isinstance(metadata,dict) or 'working_geometry' not in metadata:raise ValueError('Missing working-grid metadata')
    validate_metadata(metadata,list(reversed(metadata['working_geometry']['shape'])))
    pair=PairedSiren(3,**payload['model']).to(device)
    pair.load_state_dict(payload['state'],strict=True)
    pair.requires_working_grid_metadata=True
    pair.working_grid_metadata=metadata
    return pair.eval()


def configure_crop(branch, options):
    """An integer millimetre extent uses extent+1 voxel centers at 1 mm."""
    if 'crop_extent_mm' not in options:
        return
    size=options['crop_extent_mm']+1
    if any(n<size for n in branch.moving.shape):
        raise ValueError('Working volume is smaller than the requested physical crop')
    branch.cropped=True
    branch.set_box(branch.moving.new_zeros(3),(size,)*3)
