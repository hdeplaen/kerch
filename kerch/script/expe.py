# coding=utf-8
from ._parser import get_args, get_kwargs
from .. import data, model, train


def run(**kwargs):
    verbose = kwargs.pop('verbose')
    filename = kwargs.pop('filename')
    expe_kwargs = get_kwargs(filename)
    expe_name = filename[:-len('.yaml')]

    # LOADING DATA
    if verbose: print('Loading data...', end=" ")
    dataset = data.factory(**expe_kwargs['data'])

    # LOADING MODEL
    if verbose: print('Done'), print('Loading model...', end=" ")
    mdl = model.factory(**expe_kwargs['model'])

    # LOADING TRAINER
    if verbose: print('Done'), print('Initializing training...', end=" ")
    trainer = train.Trainer(model=mdl, learning_set=dataset, verbose=verbose, expe_name=expe_name,
                            **expe_kwargs['train'])

    # TRAIN
    if verbose: print('Done'), print('Start training...')
    trainer.fit()
    if verbose: print('Model trained')


if __name__ == '__main__':
    kwargs = get_args()
    run(**kwargs)
