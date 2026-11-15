"""Native per-call CUDA RNG inside complete-step graphs, with exact offset replay."""
import math
import torch
from medical.models.siren import coordinates_from_indices
from medical.registration.query_geometry import normal_count,query_directions


class NativeGraphInputs:
    def __init__(self,branch,count,seed,steps):
        self.branch,self.count,self.seed=branch,count,seed
        self.shape=tuple(branch.moving.shape)
        self.ranges=torch.tensor([n-c+1 for n,c in zip(self.shape,branch.crop_shape)],device=branch.moving.device)
        self.half=branch.moving.new_tensor([c-1 for c in branch.crop_shape])/2
        self.crop_volume=math.prod(branch.crop_shape)
        self.seeds=[seed,seed+4000037,seed+4000047,seed+4000057]
        self.offsets=self.measure_offsets()
        self.groups={tag:self.generators() for tag in range(max(1,len(branch.crop_center_sites)))}
        self.choices=[0]*steps
        if branch.crop_center_sites:
            generator=self.generator(seed)
            choices=torch.empty(steps,device=branch.moving.device,dtype=torch.long)
            for step in range(steps):
                generator.set_offset(step*self.offsets[0])
                choices[step:step+1].copy_(torch.randint(len(branch.crop_center_sites),(1,),
                    device=branch.moving.device,generator=generator))
            self.choices=choices.cpu().tolist()
        self.early=branch.moving.new_tensor(.25);self.late=branch.moving.new_tensor(.75)

    def generator(self,seed):
        return torch.Generator(device=self.branch.moving.device).manual_seed(seed)

    def generators(self):return [self.generator(seed) for seed in self.seeds]

    def measure_offsets(self):
        branch=self.branch;device=branch.moving.device
        source_offsets=[]
        for tag in range(max(1,len(branch.crop_center_sites))):
            generator=self.generator(self.seed)
            if branch.crop_center_sites:
                torch.randint(len(branch.crop_center_sites),(1,),device=device,generator=generator)
                torch.randint(len(branch.crop_center_sites[tag]),(1,),device=device,generator=generator)
            else:torch.rand(3,device=device,generator=generator)
            torch.randint(self.crop_volume,(self.count,),device=device,generator=generator)
            source_offsets.append(generator.get_offset())
        if len(set(source_offsets))!=1:raise ValueError('Endpoint mask bounds changed the CUDA RNG counter layout')
        particle,query,time=self.generators()[1:]
        for _ in range(2):torch.rand(branch.particles,3,device=device,generator=particle)
        torch.randn(len(branch.centres),normal_count(branch.pairs_per_cell,branch.query_geometry),3,device=device,generator=query)
        if branch.random_timesteps:
            for _ in range(2):torch.rand((),device=device,generator=time)
        return [source_offsets[0],particle.get_offset(),query.get_offset(),time.get_offset()]

    def reset_offsets(self,tag,step):
        for generator,offset in zip(self.groups[tag],self.offsets):generator.set_offset((step-1)*offset)

    def draw(self,tag):
        branch=self.branch;device=branch.moving.device
        source,particle,query,time=self.groups[tag]
        if branch.crop_center_sites:
            endpoint=torch.randint(len(branch.crop_center_sites),(1,),device=device,generator=source)
            torch._assert_async((endpoint==tag).all(),'Native crop endpoint RNG is out of sequence')
            sites=branch.crop_center_sites[tag]
            index=torch.randint(len(sites),(1,),device=device,generator=source)
            center=sites[index].squeeze(0)
            origin=torch.minimum((center-self.half).round().long().clamp_min(0),self.ranges-1)
        else:origin=(torch.rand(3,device=device,generator=source)*self.ranges).long()
        xyz_origin=origin.flip(0).to(branch.moving.dtype)
        local=torch.randint(self.crop_volume,(self.count,),device=device,generator=source)
        indices=torch.zeros_like(local)
        local_stride=global_stride=1
        for axis in range(2,-1,-1):
            component=torch.div(local,local_stride,rounding_mode='floor')%branch.crop_shape[axis]
            indices+=(component+origin[axis])*global_stride
            local_stride*=branch.crop_shape[axis];global_stride*=self.shape[axis]
        coordinates=coordinates_from_indices(indices,self.shape).to(branch.moving.dtype)
        margin=(1-branch.particle_sampling_fraction)/2
        low=xyz_origin+margin*branch.domain_extent
        size=branch.particle_sampling_fraction*branch.domain_extent
        positions=tuple(low+torch.rand(branch.particles,3,device=device,dtype=branch.moving.dtype,
                        generator=particle)*size for _ in range(2))
        directions=torch.randn(len(branch.centres),normal_count(branch.pairs_per_cell,branch.query_geometry),3,device=device,
                               dtype=branch.moving.dtype,generator=query)
        directions=query_directions(directions,branch.pairs_per_cell,branch.query_geometry)
        offsets=branch.pair_radius_mm*directions/branch.spacing_xyz
        centers=branch.local_centres+xyz_origin
        sites=centers[:,None,None]+torch.stack((offsets,-offsets),-2)
        queries=2*sites/branch.extent-1
        early,late=self.early,self.late
        if branch.random_timesteps:
            early=.5*torch.rand((),device=device,dtype=branch.moving.dtype,generator=time)
            late=.5+.5*torch.rand((),device=device,dtype=branch.moving.dtype,generator=time)
        return {'coordinates':coordinates,'source_truth':branch.moving.flatten()[indices],
                'target_truth':branch.fixed.flatten()[indices],'forward_positions':positions[0],
                'reverse_positions':positions[1],'queries':queries,'origin':xyz_origin,'early':early,'late':late}
