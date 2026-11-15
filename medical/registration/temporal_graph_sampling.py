"""Scrambled temporal quadrature with unchanged native spatial random streams."""
import torch

from medical.registration.native_graph_sampling import NativeGraphInputs


class SobolTimeInputs(NativeGraphInputs):
    def __init__(self,branch,count,seed,steps):
        super().__init__(branch,count,seed,steps)
        self.table=torch.quasirandom.SobolEngine(2,scramble=True,seed=seed+4000057).draw(steps).to(branch.moving.device)
        self.step_index=torch.zeros(1,device=branch.moving.device,dtype=torch.long)

    def reset_offsets(self,tag,step):
        super().reset_offsets(tag,step)
        self.step_index.fill_(step-1)

    def draw(self,tag):
        data=super().draw(tag)
        if self.branch.random_timesteps:
            point=self.table.index_select(0,self.step_index).reshape(2)
            data['early']=.5*point[0]
            data['late']=.5+.5*point[1]
        return data
