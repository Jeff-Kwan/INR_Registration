"""Single external-baseline case with a recorded seed override."""
import argparse
import importlib
import json
from pathlib import Path
from medical.case_runner import run_case
from .sources import SOURCES
from .common import validate_config


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method',choices=SOURCES,required=True)
    for name in ('moving','fixed','config','output'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--seed',type=int)
    args=parser.parse_args(argv)
    config=json.loads(Path(args.config).read_text())
    if args.seed is not None:
        if not 0 <= args.seed < 2**32: parser.error('Seed must be in [0,2**32)')
        config['seed']=args.seed
    validate_config(config,args.method)
    method=importlib.import_module('baselines.'+args.method).run_case
    scores=run_case(args.moving,args.fixed,config,args.output,method,args.device)
    print(json.dumps(scores,indent=2,allow_nan=False))


if __name__=='__main__': main()
