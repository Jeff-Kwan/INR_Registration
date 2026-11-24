"""Native per-call CUDA RNG inside complete-step graphs, with exact offset replay."""
import math
import torch
from medical.models.siren import coordinates_from_indices
from medical.registration.query_geometry import normal_count,query_directions
from medical.registration.training_integrators import sample_times


class NativeGraphInputs:
    def __init__(self, branch, count, seed):
        self.branch, self.count = branch, count
        self.shape=tuple(branch.moving.shape)
        self.ranges=torch.tensor([n-c+1 for n,c in zip(self.shape,branch.crop_shape)],device=branch.moving.device)
        self.crop_volume=math.prod(branch.crop_shape)
        self.seeds=[seed,seed+4000037,seed+4000047,seed+4000057]
        self.offsets = self.measure_offsets()
        self.streams = self.generators()

    def generator(self,seed):
        return torch.Generator(device=self.branch.moving.device).manual_seed(seed)

    def generators(self):return [self.generator(seed) for seed in self.seeds]

    def measure_offsets(self):
        branch=self.branch;device=branch.moving.device
        source, particle, query, time = self.generators()
        torch.rand(3, device=device, generator=source)
        torch.randint(self.crop_volume, (self.count,), device=device, generator=source)
        for _ in range(2):torch.rand(branch.particles,3,device=device,generator=particle)
        torch.randn(len(branch.centres),normal_count(branch.pairs_per_cell,branch.query_geometry),3,device=device,generator=query)
        if branch.random_timesteps:
            for _ in range(branch.time_strata):torch.rand((),device=device,generator=time)
        return [stream.get_offset() for stream in (source, particle, query, time)]

    def reset_offsets(self, step):
        for generator, offset in zip(self.streams, self.offsets):
            generator.set_offset((step - 1) * offset)

    def draw(self):
        branch = self.branch
        device = branch.moving.device
        source, particle, query, time = self.streams
        origin = (torch.rand(3, device=device, generator=source) * self.ranges).long()
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
        times=sample_times(branch.moving,branch.time_strata,branch.random_timesteps,time)
        return {'coordinates':coordinates,'source_truth':branch.moving.flatten()[indices],
                'target_truth':branch.fixed.flatten()[indices],'forward_positions':positions[0],
                'reverse_positions':positions[1],'queries':queries,'origin':xyz_origin,'times':times}
