# coding=utf-8
from ._parser import get_args
from . import version, expe


def run():
    kwargs = get_args()
    action_name = kwargs.pop('action')[0]

    switcher = {'expe': expe}
    print(action_name)
    script = switcher.get(action_name)
    return script.run(**kwargs)


if __name__ == '__main__':
    run()
