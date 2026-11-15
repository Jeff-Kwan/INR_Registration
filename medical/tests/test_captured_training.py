"""Complete-objective VJPs and exact sampler replay; no optimizer steps or artifacts."""
import copy
import json
from pathlib import Path
import unittest

import torch
from medical.configuration import load_config,validate_config
from medical.models.siren import PairedSiren,coordinates_from_indices
from medical.registration.captured_objective import joint_loss
from medical.registration.native_graph_sampling import NativeGraphInputs
from medical.registration.temporal_graph_sampling import SobolTimeInputs
from medical.registration.ncc import ncc_options
from medical.registration.objective import TrainingBatch,endpoint_loss
from medical.registration.query_geometry import normal_count,query_directions
from medical.registration.symmetric_registration import SymmetricPairRegistration,sample_crop_indices


class TestCapturedTraining(unittest.TestCase):
    def test_api_default_matches_current_native_recipe(self):
        path=Path(__file__).parents[1]/'configs/default.json'
        actual=load_config();self.assertEqual(actual,load_config(path))
        self.assertEqual(actual['objective']['endpoint'],'ncc')
        self.assertEqual(actual['objective']['registration']['particles'],32768)
        actual['objective']['registration']['particles']=1
        self.assertEqual(load_config()['objective']['registration']['particles'],32768)

    def test_orthogonal_frames_and_default_directions(self):
        normals=torch.randn(97,2,3,generator=torch.Generator().manual_seed(42),dtype=torch.float64)
        expected=normals/normals.norm(dim=-1,keepdim=True).clamp_min(1e-12)
        torch.testing.assert_close(query_directions(normals,2,'independent'),expected,rtol=0,atol=0)
        for pairs in (2,3):
            actual=query_directions(normals,pairs,'orthogonal')
            torch.testing.assert_close(actual@actual.transpose(-1,-2),torch.eye(pairs).double().expand(97,-1,-1),rtol=1e-12,atol=1e-12)
            self.assertEqual(normal_count(pairs,'orthogonal'),2)
        for pairs,geometry in [(1,'orthogonal'),(4,'orthogonal'),(2,'unknown')]:
            with self.assertRaises(ValueError):normal_count(pairs,geometry)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA native-generator replay check')
    def test_native_sampling_matches_reference_inside_graphs(self):
        device=torch.device('cuda:0');image=torch.arange(17*19*23,device=device,dtype=torch.float32).reshape(17,19,23)
        for masked in (False,True):
            for geometry in ('independent','orthogonal'):
                branch=SymmetricPairRegistration(image,image.flip(0),torch.eye(4),grid_size=4,particles=31,
                    pairs_per_cell=2,crop_fraction=.5,random_timesteps=True,query_geometry=geometry,
                    compile_derivatives=False,compile_velocity=False,gaussian_kernel='three_tap')
                branch.crop_center_sites=([((image%2==0).nonzero().float()),((image%3==0).nonzero().float())] if masked else [])
                generator=torch.Generator(device=device).manual_seed(42);expected=[]
                for step in range(12):
                    indices=sample_crop_indices(branch,47,generator);positions=branch.sample_positions();branch.resample_queries();early,late=branch._sample_times()
                    x=coordinates_from_indices(indices,image.shape)
                    expected.append({k:v.clone() for k,v in dict(coordinates=x,source_truth=image.flatten()[indices],
                        target_truth=branch.fixed.flatten()[indices],forward_positions=positions[0],reverse_positions=positions[1],
                        queries=branch.coordinates,origin=branch.origin,early=early,late=late).items()})
                sampler=NativeGraphInputs(branch,47,42,12);graphs={};outputs={}
                for step in range(1,13):
                    tag=sampler.choices[step-1];sampler.reset_offsets(tag,step)
                    if tag not in graphs:
                        graph=torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):outputs[tag]=sampler.draw(tag)
                        graphs[tag]=graph;sampler.reset_offsets(tag,step)
                    graphs[tag].replay()
                    for name,value in expected[step-1].items():
                        torch.testing.assert_close(outputs[tag][name],value,rtol=0,atol=0)

    def test_complete_losses_and_all_parameter_gradients(self):
        torch.manual_seed(42);image=torch.rand(17,19,23,dtype=torch.float64)
        pair=PairedSiren(3,width=8,layers=3,omega=10.).double()
        with torch.no_grad():
            for p in pair.target.parameters():p.add_(torch.randn_like(p)*.001)
        for similarity in ('ncc','gw_ncc'):
            branch=SymmetricPairRegistration(image,image.flip(0),torch.eye(4),grid_size=4,particles=257,
                pairs_per_cell=2,crop_fraction=.75,random_timesteps=False,compile_derivatives=False,
                compile_velocity=False,gaussian_kernel='three_tap',registration_similarity=similarity)
            indices=sample_crop_indices(branch,61,torch.Generator().manual_seed(42));x=coordinates_from_indices(indices,image.shape).double()
            batch=TrainingBatch(x,image.flatten()[indices],branch.fixed.flatten()[indices],x)
            positions=branch.sample_positions();registration=branch.loss(pair,positions=positions);endpoints=pair.prepare(temporal=False)
            for weight,coefficient in ((.1,.01),(0.,.01),(.1,0.),(0.,0.)):
                settings={'endpoint':'ncc','ncc':{'moment_weight':.01}}
                reference=endpoint_loss(pair,batch,settings)['total']+weight*registration['grid_pdlk']+coefficient*registration['registration']
                actual,finite=joint_loss(endpoints.source,endpoints.target,endpoints.omegas,x,x,batch.source,batch.target,
                    branch.coordinates,branch.spatial_scale,positions,branch.moving,branch.fixed,branch.origin,
                    branch.domain_extent,branch.extent,image.new_tensor(.25),image.new_tensor(.75),tuple(branch.sigma),
                    branch.grid_size,branch.c,branch.eta,weight,coefficient,ncc_options(settings['ncc']),
                    {'registration_similarity':similarity})
                self.assertTrue(finite);torch.testing.assert_close(actual['total'],reference,rtol=1e-9,atol=1e-10)
                ga=torch.autograd.grad(actual['total'],tuple(pair.parameters()),retain_graph=True)
                gb=torch.autograd.grad(reference,tuple(pair.parameters()),retain_graph=True)
                for a,b in zip(ga,gb):torch.testing.assert_close(a,b,rtol=1e-8,atol=1e-10)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA Sobol sampler replay check')
    def test_sobol_times_replay_without_changing_spatial_samples(self):
        device=torch.device('cuda:0');image=torch.arange(17*19*23,device=device,dtype=torch.float32).reshape(17,19,23)
        branch=SymmetricPairRegistration(image,image.flip(0),torch.eye(4),grid_size=4,particles=31,
            pairs_per_cell=2,crop_fraction=.5,random_timesteps=True,query_geometry='orthogonal',
            compile_derivatives=False,compile_velocity=False,gaussian_kernel='three_tap')
        branch.crop_center_sites=[(image%2==0).nonzero().float(),(image%3==0).nonzero().float()]
        native=NativeGraphInputs(branch,47,42,12);sobol=SobolTimeInputs(branch,47,42,12)
        self.assertEqual(native.choices,sobol.choices)
        times=torch.quasirandom.SobolEngine(2,scramble=True,seed=42+4000057).draw(12).to(device)
        graphs={};outputs={}
        for step in range(1,13):
            tag=sobol.choices[step-1];native.reset_offsets(tag,step);expected=native.draw(tag)
            sobol.reset_offsets(tag,step)
            if tag not in graphs:
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):outputs[tag]=sobol.draw(tag)
                graphs[tag]=graph;sobol.reset_offsets(tag,step)
            graphs[tag].replay()
            for key,value in expected.items():
                if key not in ('early','late'):torch.testing.assert_close(outputs[tag][key],value,rtol=0,atol=0)
            torch.testing.assert_close(outputs[tag]['early'],.5*times[step-1,0],rtol=0,atol=0)
            torch.testing.assert_close(outputs[tag]['late'],.5+.5*times[step-1,1],rtol=0,atol=0)

    def test_explicit_preset_and_invalid_runtime_options(self):
        path=Path(__file__).parents[1]/'configs/native_siren_captured_5k.json'
        config=json.loads(path.read_text());validate_config(config)
        for value in ('unknown',False):
            candidate=copy.deepcopy(config);candidate['runtime']['training_execution']=value
            with self.assertRaises(ValueError):validate_config(candidate)
        for cap in (0,1,-1,2.5,True):
            candidate=copy.deepcopy(config);candidate['runtime']['recovery_max_axis']=cap
            with self.assertRaises(ValueError):validate_config(candidate)
        candidate=copy.deepcopy(config);candidate['spatial_units']='mm'
        with self.assertRaises(ValueError):validate_config(candidate)
        candidate=copy.deepcopy(config);candidate['objective']['registration'].update(query_geometry='orthogonal',pairs_per_cell=4)
        with self.assertRaises(ValueError):validate_config(candidate)
        candidate=copy.deepcopy(config);candidate['training']['temporal_sampling']='sobol';validate_config(candidate)
        for mode in ('unknown',False):
            candidate=copy.deepcopy(config);candidate['training']['temporal_sampling']=mode
            with self.assertRaises(ValueError):validate_config(candidate)
        for section,key,value in [('runtime','training_execution','standard'),('objective','registration',None)]:
            candidate=copy.deepcopy(config);candidate['training']['temporal_sampling']='sobol'
            if section=='runtime':candidate[section][key]=value
            else:candidate['objective']['registration']['random_timesteps']=False
            with self.assertRaises(ValueError):validate_config(candidate)
