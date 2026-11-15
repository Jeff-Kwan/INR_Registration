"""Full-gradient native NCC/PD-LK/registration training with a captured AdamW step."""
import json
import torch
from medical.models.siren import PairedSiren
from medical.models.siren_functional import siren_parameters
from medical.registration.symmetric_registration import SymmetricPairRegistration
from medical.registration.objective import scheduled_weight
from medical.registration.optimization import make_optimizers,learning_rate_factor,objective_factor,registration_weight
from medical.registration.ncc import ncc_options
from medical.registration.captured_objective import joint_loss
from medical.registration.native_graph_sampling import NativeGraphInputs


_LAST_COMPILE_SIGNATURE = None


def fit_pair(moving,fixed,config,*,report=None,affine=None,moving_mask=None,fixed_mask=None,checkpoint_callback=None):
    from medical.configuration import validate_config
    validate_config(config)
    settings,objective=config['training'],config['objective']
    if not moving.is_cuda or moving.dtype!=torch.float32 or settings.get('optimizer',{}).get('name')!='adamw':
        raise ValueError('The complete-graph experiment requires CUDA FP32 AdamW')
    if ('working_grid' in config or config.get('spatial_units','voxel')!='voxel' or objective['endpoint']!='ncc'
        or objective.get('regularizer','pdlk')!='pdlk'):
        raise ValueError('The complete-graph experiment uses the ordinary native NCC/PD-LK objective')
    if any(k in settings for k in ['endpoint_sampling','structure_crop_sampling','final_crop_fraction']) or settings.get('tie_registration_points',False):
        raise ValueError('The complete-graph experiment keeps fixed ordinary shared crops and separate particles')
    if objective['registration']['registration_similarity'] not in ('ncc','gw_ncc') or objective['registration']['gaussian_kernel']!='three_tap':
        raise ValueError('Captured execution requires box or GW-NCC and three-tap velocity smoothing')
    if objective['registration']['registration_similarity']=='gw_ncc' and objective['registration']['gw_ncc_kernel']!='three_tap':
        raise ValueError('Captured GW-NCC uses the retained three-tap Gaussian')
    if not (config['runtime']['compile_derivatives'] and config['runtime']['compile_endpoints']
            and objective['registration']['compile_velocity'] and objective['registration'].get('compile_decoder',False)):
        raise ValueError('Captured execution requires compiled endpoints, derivatives, velocity and decoder')
    if objective['registration']['recompute'] or config['model'].get('first_omega_mode')=='parameterized':
        raise ValueError('Captured execution requires ordinary parameters without recomputation')
    torch.manual_seed(config['runtime']['seed'])
    torch.set_float32_matmul_precision('highest')
    pair=PairedSiren(3,**config['model']).to(moving.device)
    pair.training_metadata={'regularizer':'pdlk','regularizer_weight':objective['weight'],
        'endpoint_objective':'ncc','endpoint_sampling':'shared_registration_crop','spatial_units':'voxel',
        'training_integrator':'two_half_steps_spatial_midpoint_rk2','mask_role':'crop_center_only',
        'execution':'captured',
        'matmul_precision':torch.get_float32_matmul_precision(),
        'sampling':'exact native per-call CUDA RNG streams',
        'query_geometry':objective['registration'].get('query_geometry','independent'),
        'registration_similarity':objective['registration']['registration_similarity'],
        'optimization':{'optimizer':settings['optimizer'],'lr_schedule':settings.get('lr_schedule',{'name':'exponential'}),
                        'first_layer_lr_factor':settings.get('first_layer_lr_factor',1.)}}
    temporal_sampling=settings.get('temporal_sampling','independent')
    pair.training_metadata['temporal_sampling']=temporal_sampling
    if temporal_sampling=='sobol':
        pair.training_metadata['sampling']='native spatial CUDA RNG; scrambled2D Sobol temporal quadrature'
    branch=SymmetricPairRegistration(moving,fixed,affine,seed=config['runtime']['seed'],c=objective['c'],eta=objective['eta'],
        compile_derivatives=True,**{k:v for k,v in objective['registration'].items() if k!='weight'})
    if not branch.cropped:raise ValueError('A fixed crop is required')
    branch.crop_center_sites=[]
    for image,mask in [(moving,moving_mask),(fixed,fixed_mask)]:
        if mask is not None:
            if mask.shape!=image.shape or mask.dtype!=torch.bool or mask.device!=image.device:raise ValueError('Invalid crop-center mask')
            sites=mask.nonzero().to(image.dtype)
            if not len(sites):raise ValueError('Empty crop-center mask')
            branch.crop_center_sites.append(sites)
    ordinary=make_optimizers(pair,settings)
    assert len(ordinary)==1
    base_rates=[g['convergence_base_lr'] for g in ordinary[0].param_groups]
    learning_rates=[torch.tensor(rate,device=moving.device) for rate in base_rates]
    groups=[{'params':g['params'],'lr':lr} for g,lr in zip(ordinary[0].param_groups,learning_rates)]
    options=settings['optimizer']
    optimizer=torch.optim.AdamW(groups,lr=learning_rates[0],betas=tuple(options.get('betas',[.9,.999])),
        weight_decay=options.get('weight_decay',0.),fused=True,capturable=True)
    del ordinary
    input_class=NativeGraphInputs
    if temporal_sampling=='sobol':
        from medical.registration.temporal_graph_sampling import SobolTimeInputs
        input_class=SobolTimeInputs
    inputs=input_class(branch,settings['batch_size'],config['runtime']['seed'],settings['steps'])
    source_parameters,omegas=siren_parameters(pair.source)
    target_parameters,target_omegas=siren_parameters(pair.target)
    if omegas!=target_omegas:raise ValueError('Endpoint frequencies must match')
    pdlk_weight=torch.zeros((),device=moving.device)
    registration_coefficient=torch.zeros((),device=moving.device)
    # Reuse code for adjacent equal geometries with fresh tensors/optimizer state.
    # Retire prior specializations on change to bound the worker's shape cache.
    global _LAST_COMPILE_SIGNATURE
    signature=(tuple(moving.shape),str(moving.dtype),str(moving.device),json.dumps(config,sort_keys=True))
    if signature != _LAST_COMPILE_SIGNATURE:
        torch._dynamo.reset_code(joint_loss.__code__)
        _LAST_COMPILE_SIGNATURE=signature
    compiled=torch.compile(joint_loss,fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
    options_ncc=ncc_options(objective.get('ncc'))
    def update(tag):
        data=inputs.draw(tag)
        pair.zero_grad(set_to_none=False)
        terms,finite=compiled(source_parameters,target_parameters,omegas,
            data['coordinates'],data['coordinates'],data['source_truth'],data['target_truth'],data['queries'],
            branch.spatial_scale,(data['forward_positions'],data['reverse_positions']),moving,fixed,
            data['origin'],branch.domain_extent,branch.extent,data['early'],data['late'],tuple(branch.sigma),
            branch.grid_size,branch.c,branch.eta,pdlk_weight,registration_coefficient,options_ncc,objective['registration'])
        torch._assert_async(finite & torch.isfinite(terms['total']),'Nonfinite captured training objective or trajectory')
        terms['total'].backward()
        norm=None
        if settings['grad_clip']:
            norm=torch.nn.utils.clip_grad_norm_(pair.parameters(),settings['grad_clip'],error_if_nonfinite=False,foreach=True)
            torch._assert_async(torch.isfinite(norm),'Nonfinite captured training gradient')
        optimizer.step()
        return terms,norm
    stream=torch.cuda.Stream(device=moving.device)
    stream.wait_stream(torch.cuda.current_stream(moving.device))
    graphs={}
    captured_outputs={}
    warmup=min(3,settings['steps'])
    with torch.cuda.stream(stream):
        for step in range(1,settings['steps']+1):
            tag=inputs.choices[step-1]
            inputs.reset_offsets(tag,step)
            lr_factor=learning_rate_factor(settings,step)
            for tensor,base in zip(learning_rates,base_rates):tensor.fill_(base*lr_factor)
            activation=objective_factor(settings,step)
            weight=scheduled_weight(objective,step,settings['steps'])*activation
            registration_weight_value=registration_weight(settings,objective['registration']['weight'],step)*activation
            pdlk_weight.fill_(weight);registration_coefficient.fill_(registration_weight_value)
            if step<=warmup:
                terms,norm=update(tag)
            else:
                if tag not in graphs:
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph,stream=stream):captured_outputs[tag]=update(tag)
                    graphs[tag]=graph
                    inputs.reset_offsets(tag,step)
                graphs[tag].replay()
                terms,norm=captured_outputs[tag]
            if checkpoint_callback is not None:checkpoint_callback(step,pair)
            if report is not None and (step%settings['log_every']==0 or step==settings['steps']):
                report({'step':step,**{k:float(v.detach()) for k,v in terms.items()},
                        'lr':settings['lr']*lr_factor,
                        'first_layer_lr':settings['lr']*lr_factor*settings.get('first_layer_lr_factor',1.),
                        'gradient_norm_before_clip':None if norm is None else float(norm.detach()),
                        'effective_pdlk_weight':weight,'effective_registration_weight':registration_weight_value})
        stream.synchronize()
    torch.cuda.current_stream(moving.device).wait_stream(stream)
    pair.zero_grad(set_to_none=True)
    return pair.eval()
