"""Our paired method on official Learn2Reg OASIS public validation."""
from medical.abdomen_method import run_case as run_prepared_case

def run_case(case,config,device,*,checkpoint_callback=None):
    return run_prepared_case(case,config,device,protocol="oasis",checkpoint_callback=checkpoint_callback)
