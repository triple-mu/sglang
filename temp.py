import torch
import torch.distributed._functional_collectives as fc
import torch.distributed._functional_collectives_impl as fci
fci._all_to_all_single()

fc.all_to_all_single()

torch.distributed.barrier()

torch.distributed.all_to_all_single
