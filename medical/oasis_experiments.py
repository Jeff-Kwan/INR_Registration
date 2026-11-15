"""Plan official OASIS validation jobs; --run is required for execution."""
from medical.abdomen_experiments import main as shared_main

def main(argv=None):
    return shared_main(argv,dataset="oasis")

if __name__ == "__main__": main()
