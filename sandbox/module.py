# --- hide: start ---
import kerch
import torch
from typing import Iterator

torch.manual_seed(0)


class MyModule(kerch.feature.Module):
    def __init__(self, *args, **kwargs):
        super(MyModule, self).__init__(*args, **kwargs)

        # we recover the parameter size by the argument param_size
        param_size = kwargs.pop('param_size', (1, 1))

        # we create our parameter of float type kerch.FTYPE
        # (this value can be modified and ensures that all floating types are the same throughout the code)
        self.my_param = torch.nn.Parameter(torch.randn(param_size, dtype=kerch.FTYPE), requires_grad=True)

    def _euclidean_parameters(self, recurse=True) -> Iterator[torch.nn.Parameter]:
        # important not to forget, otherwise the parameters returned by mother classes will be skipped
        yield from super(MyModule, self)._euclidean_parameters(recurse=recurse)

        # we yield our additional new parameter
        yield self.my_param

    def after_step(self):
        # after each training step, we want the rows to be centered
        with torch.no_grad():
            self.my_param.data = self.my_param - torch.mean(self.my_param, dim=0)

    @property
    def hparams_fixed(self) -> dict:
        # we add the shape of our parameter to the fixed hyperparameters
        # we don't forget to return the other possible hyperparameters issued by parent classes
        return {'my_param size': self.my_param.shape,
                **super(MyModule, self).hparams_fixed}


my_module = MyModule(param_size=(2, 3))

# --- hide: stop ---

# we suppose that an optimization step has been performed
my_module.after_step()

# Euclidean parameters
print('EUCLIDEAN PARAMETERS:')
for p in my_module.manifold_parameters(type='euclidean'):
    print(p)
