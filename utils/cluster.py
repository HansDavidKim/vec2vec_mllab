### Namespace class is for storing result of parsing input command
### e.g) --epoch 10
from argparse import Namespace

### datasets library for fetching datasets in Hugging Face
import datasets

'''
Wrapper class for CDE Dataset.

In this way, we could cluster dataset with respect to 
sub-domain and enhances quality of batch sampling.
'''
class CdeDatasetWrapper(datasets.Dataset):
    ### Type Hint for dataset
    dataset: datasets.Dataset
    def __init__(
            self, 
            dataset: datasets.Dataset,
        ):
        from cde.dataset import get_subdomain_idxs_cached
        self.dataset = dataset
        self.subdomain_idxs = get_subdomain_idxs_cached(self.dataset)
        self._fingerprint = dataset._fingerprint

    
    def __len__(self):
        return len(self.dataset)

### Sample data according to config (cfg)
def make_cluster_sampler(dset: datasets.Dataset, cfg: Namespace):
    from cde.sampler import AutoClusterWithinDomainSampler

    dset_wrapper = CdeDatasetWrapper(dset)
    return AutoClusterWithinDomainSampler(
        dataset=dset_wrapper,
        query_to_doc=True,
        batch_size=2048,
        cluster_size=cfg.bs,
        shuffle=True,
        share_negatives_between_gpus=False,
        downscale_and_normalize=True,
        # model="gtr_base",
        model="bert",
        seed=cfg.seed,
    )