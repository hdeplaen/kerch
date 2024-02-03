===============
Kernel Smoother
===============

Functions
=========

Coefficients-Based Smoother
---------------------------

.. autofunction:: kerch.method.smoother


Kernel Smoother
---------------

.. autofunction:: kerch.method.kernel_smoother


Example
=======

Different Kernels, Same Bandwidth
---------------------------------

As we can see in the following example, different kernels have different behaviors. They all use the same bandwidth as
without specification, the bandwidth is based on the distance matrix which is here the same for all as they are all
based on the same domain. The same bandwidth is not always appropriate for different kernels.

.. plot::
    :include-source:

    import kerch
    import torch
    from matplotlib import pyplot as plt

    # data
    fun = lambda x: torch.sin(x ** 2)

    x_equal = torch.linspace(0, 2, 100)
    x_nonequal = 2 * torch.sort(torch.rand(40)).values

    y_original = fun(x_equal)
    y_noisy = fun(x_nonequal) + .2 * torch.randn_like(x_nonequal)

    plt.plot(x_equal, y_original, label="Original Data", color="black", linestyle='dotted')
    plt.scatter(x_nonequal, y_noisy, label="Noisy Data", color="black")

    # kernels
    kernels = [('RBF', 'red'),
               ('Laplacian', 'orange'),
               ('Logistic', 'olive'),
               ('Epanechnikov', 'gold'),
               ('Quartic', 'chartreuse'),
               ('Silverman', 'green'),
               ('Triangular', 'teal'),
               ('Tricube', 'cyan'),
               ('Triweight', 'royalblue'),
               ('Uniform', 'purple')]

    # kernel smoother
    for name, c in kernels:
        y_reconstructed = kerch.method.kernel_smoother(domain=x_nonequal, observations=y_noisy, kernel_type=name.lower())
        plt.plot(x_nonequal, y_reconstructed, label=name, color=c)

    # plot
    plt.title('Kernel Smoothing')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='lower center', ncol=3)


Same Kernels, Different Bandwidths
----------------------------------

In this example, we show how two different kernels react differently based on the prescribed bandwidth. We will
consider two kernels, the :class:`~kerch.kernel.Laplacian` with a very heavy tail and the very restricted
:class:`~kerch.kernel.Triweight`. We can first have a view at their respective shapes.

.. plot::
    :include-source:

    import kerch
    import torch
    from matplotlib import pyplot as plt

    # domain
    x = torch.linspace(-3, 3, 500)

    # define the kernels
    k_l1 = kerch.kernel.Laplacian(sample=x, sigma=1)
    k_l2 = kerch.kernel.Laplacian(sample=x, sigma=2)
    k_t1 = kerch.kernel.Triweight(sample=x, sigma=1)
    k_t2 = kerch.kernel.Triweight(sample=x, sigma=2)

    # plot the shapes
    plt.plot(x, k_l1.k(y=0).squeeze(), label=f"Laplacian with $\sigma$={k_l1.sigma}", color='black')
    plt.plot(x, k_l2.k(y=0).squeeze(), label=f"Laplacian with $\sigma$={k_l2.sigma}", color='black', linestyle='dashed')
    plt.plot(x, k_t1.k(y=0).squeeze(), label=f"Triweight with $\sigma$={k_t1.sigma}", color='red')
    plt.plot(x, k_t2.k(y=0).squeeze(), label=f"Triweight with $\sigma$={k_t2.sigma}", color='red', linestyle='dashed')

    # annotate the plot
    plt.title('Kernel Shape')
    plt.xlabel('x')
    plt.ylabel('k(x,y=0)')
    plt.ylim(-.25, 1.1)
    plt.legend(loc='lower center', ncol=2)

.. plot::
    :include-source:

    import kerch
    import torch
    from matplotlib import pyplot as plt

    # data
    fun = lambda x: torch.sin(x ** 2)

    x_equal = torch.linspace(0, 2, 100)
    x_nonequal = 2 * torch.sort(torch.rand(40)).values

    y_original = fun(x_equal)
    y_noisy = fun(x_nonequal) + .2 * torch.randn_like(x_nonequal)

    # plot
    fig, axs = plt.subplots(1, 2)
    for ax in axs.flatten():
        ax.plot(x_equal, y_original, label="Original Data", color="black", linestyle='dotted')
        ax.scatter(x_nonequal, y_noisy, label="Noisy Data", color="black")
        plt.title('Kernel Smoothing')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    # kernel smoother
    sigmas = [(0.05, 'red'),
              (0.2, 'green'),
              (0.5, 'cyan'),
              (1.0, 'purple')]
    for s, c in sigmas:
        y_laplacian = kerch.method.kernel_smoother(domain=x_nonequal, observations=y_noisy, kernel_type='laplacian', sigma=s)
        y_triweight = kerch.method.kernel_smoother(domain=x_nonequal, observations=y_noisy, kernel_type='triweight', sigma=s)
        axs[0].plot(x_nonequal, y_laplacian, color=c, label=f"Bandwidth $\sigma$={s}")
        axs[1].plot(x_nonequal, y_triweight, color=c)

    # plot
    fig.suptitle('Kernel Smoothing')
    axs[0].set_title('Laplacian')
    axs[1].set_title('Triweight')
    fig.legend(*axs[0].get_legend_handles_labels(), loc='lower center', ncol=3)
