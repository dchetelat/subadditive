import ecole
from pathlib import Path

def create_instances(instances, folder):
    print(f"Generating {folder}")
    folder.mkdir(parents=True, exist_ok=True)
    for instance_count in range(100):
        instance = next(instances)
        instance.write_problem(str(folder/f"instance_{instance_count}.lp"))

instances = ecole.instance.SetCoverGenerator()
folder = Path("data/instances/setcover")
create_instances(instances, folder)

instances = ecole.instance.CombinatorialAuctionGenerator()
folder = Path("data/instances/cauctions")
create_instances(instances, folder)

instances = ecole.instance.IndependentSetGenerator(n_nodes=300)
folder = Path("data/instances/indset")
create_instances(instances, folder)

instances = ecole.instance.CapacitatedFacilityLocationGenerator(n_customers=20, n_facilities=10, continuous_assignment=False)
folder = Path("data/instances/facilities")
create_instances(instances, folder)
