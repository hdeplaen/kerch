import torch
import torch.autograd as autograd

class RBF(autograd.Function):

    @staticmethod
    def forward(ctx, input, weight, sigma):
        diff = (input - weight)
        norm2 = diff * diff
        fact = 1/(2*sigma^2)
        output = torch.exp(-fact*norm2)
        ctx.save_for_backward(input, output, norm2, fact)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, output, norm2, fact = ctx.saved_tensors
        grad = 2 * fact * output * torch.sqrt(norm2) * input
        return grad