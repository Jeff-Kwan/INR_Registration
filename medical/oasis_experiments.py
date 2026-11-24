"""Plan SIRENPath and baseline runs on official Learn2Reg OASIS validation pairs. Execution requires --run."""
from medical.abdomen_experiments import main as shared_main

def main(argv=None):
    return shared_main(argv,dataset="oasis")

if __name__ == "__main__": main()
