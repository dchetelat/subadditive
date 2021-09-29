import ecole
from pathlib import Path

def create_instances(instances, folder):
    print(f"Generating {folder/'train'}")
    (folder/"train").mkdir(parents=True, exist_ok=True)
    for instance_count in range(1000):
        instance = next(instances)
        instance.write_problem(str(folder/f"train/instance_{instance_count}.lp"))
    print(f"Generating {folder/'test'}")
    (folder/"test").mkdir(parents=True)
    for instance_count in range(200):
        instance = next(instances)
        instance.write_problem(str(folder/f"test/instance_{instance_count}.lp"))
        
# instances = ecole.instance.SetCoverGenerator()
# folder = Path("data/instances/setcover")
# create_instances(instances, folder)

# instances = ecole.instance.CombinatorialAuctionGenerator()
# folder = Path("data/instances/cauctions")
# create_instances(instances, folder)

instances = ecole.instance.IndependentSetGenerator(n_nodes=300)
folder = Path("data/instances/indset")
create_instances(instances, folder)

instances = ecole.instance.CapacitatedFacilityLocationGenerator(n_customers=20, n_facilities=20)
folder = Path("data/instances/facilities")
create_instances(instances, folder)
