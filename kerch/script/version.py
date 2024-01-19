# coding=utf-8
import lazy_loader
from .. import (__version__,
                __author__,
                __status__,
                __credits__,
                __date__,
                __license__,
                gpu_available)
torch = lazy_loader.load('torch')

def run():
    newline = '\n'
    if gpu_available():
        gpu = f"with GPU acceleration available (CUDA version: {torch.version.cuda}."
    else:
        gpu = f"with no GPU acceleration available."
    return (f"Kerch version: {__version__} {__status__} ({__date__}). {newline}"
            f"Author: {__author__}. Copyright: {__credits__}. License: {__license__}. {newline}"
            f"The package is using PyTorch version: {torch.__version__} " + gpu)

if __name__ == "__main__":
    print(run())
