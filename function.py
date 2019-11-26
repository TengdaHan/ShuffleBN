import torch 
from torch.autograd.function import Function


def forward_shuffle(rank, world_size):
    if rank+1 < world_size:
        return rank+1
    else:
        return 0

class ShuffleBatchNorm(Function):

    @staticmethod
    def forward(self, input, weight, bias, running_mean, running_var, eps, momentum, process_group, world_size, rank):
        input = input.contiguous()

        size = input.numel() // input.size(1)
        if size == 1:
            raise ValueError('Expected more than 1 value per channel when training, got input size {}'.format(size))
        count = torch.Tensor([size]).to(input.device)

        # calculate mean/invstd for input.
        mean, invstd = torch.batch_norm_stats(input, eps)

        count_all = torch.empty(world_size, 1, dtype=count.dtype, device=count.device)
        mean_all = torch.empty(world_size, mean.size(0), dtype=mean.dtype, device=mean.device)
        invstd_all = torch.empty(world_size, invstd.size(0), dtype=invstd.dtype, device=invstd.device)

        count_l = list(count_all.unbind(0))
        mean_l = list(mean_all.unbind(0))
        invstd_l = list(invstd_all.unbind(0))

        # using all_gather instead of all reduce so we can calculate count/mean/var in one go
        count_all_reduce = torch.distributed.all_gather(count_l, count, process_group, async_op=True)
        mean_all_reduce = torch.distributed.all_gather(mean_l, mean, process_group, async_op=True)
        invstd_all_reduce = torch.distributed.all_gather(invstd_l, invstd, process_group, async_op=True)

        # wait on the async communication to finish
        count_all_reduce.wait()
        mean_all_reduce.wait()
        invstd_all_reduce.wait()

        print('[%d]mean before shuffle:'%rank, mean_l[rank][0:4])

        # shuffle global mean & invstd
        new_rank = forward_shuffle(rank, world_size)
        count = count_l[new_rank]
        mean = mean_l[new_rank].view(1,-1)
        invstd = invstd_l[new_rank].view(1,-1)

        mean, invstd = torch.batch_norm_gather_stats(
            input,
            mean,
            invstd,
            running_mean,
            running_var,
            momentum,
            eps,
            count.long().item()
        )

        print('[%d]mean after shuffle:'%rank, mean[0:4])

        self.save_for_backward(input, weight, mean, invstd)
        self.process_group = process_group
        self.world_size = world_size
        self.rank = rank 

        # apply element-wise normalization
        out = torch.batch_norm_elemt(input, weight, bias, mean, invstd, eps)
        return out

    @staticmethod
    def backward(self, grad_output):
        grad_output = grad_output.contiguous()
        saved_input, weight, mean, invstd = self.saved_tensors
        grad_input = grad_weight = grad_bias = None
        process_group = self.process_group
        world_size = self.world_size
        rank = self.rank 

        # calculate local stats as well as grad_weight / grad_bias
        mean_dy, mean_dy_xmu, grad_weight, grad_bias = torch.batch_norm_backward_reduce(
            grad_output,
            saved_input,
            mean,
            invstd,
            weight,
            self.needs_input_grad[0],
            self.needs_input_grad[1],
            self.needs_input_grad[2]
        )

        if self.needs_input_grad[0]:
            # no need to communicate with others
            # backward pass for gradient calculation
            grad_input = torch.batch_norm_backward_elemt(
                grad_output,
                saved_input,
                mean,
                invstd,
                weight,
                mean_dy,
                mean_dy_xmu
            )

        # synchronizing of grad_weight / grad_bias is not needed as distributed
        # training would handle all reduce.
        if weight is None or not self.needs_input_grad[1]:
            grad_weight = None

        if weight is None or not self.needs_input_grad[2]:
            grad_bias = None

        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None