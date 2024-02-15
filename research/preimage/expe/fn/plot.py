import kerch
import torch
from matplotlib import pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'

x = torch.linspace(-6,6, 200)
y = torch.linspace(-6,6,200)

k1 = kerch.kernel.RFHyperbola(sample=x, alpha=.1, beta=1, gamma=1, delta=1)
k2 = kerch.kernel.RFHyperbola(sample=x, alpha=.4, beta=3, gamma=5, delta=5)
k3 = kerch.kernel.RFArcsinh(sample=x)
k4 = kerch.kernel.RFLReLU(sample=x)

y1 = k1.activation_fn(x)
y2 = k2.activation_fn(x)
y3 = k3.activation_fn(x)
y4 = k4.activation_fn(x)

x1 = k1.activation_fn_inv(y)
x2 = k2.activation_fn_inv(y)
x3 = k3.activation_fn_inv(y)
x4 = k4.activation_fn_inv(y)


fig, ax = plt.subplots()
ax.plot(x, y1, color = 'k', linestyle='solid', linewidth=2, label=f"Hyperbola (a=0.1, b=1, c=1, d=1)")
ax.plot(x, y2, color = 'k', linestyle='dotted', linewidth=2, label=f"Hyperbola (a=0.4, b=3, c=5, d=5)")
ax.plot(x, y3, color = 'k', linestyle='dashed', linewidth=2, label='Arcsinh')
ax.plot(x, y4, color = 'k', linestyle='dashdot', linewidth=2, label='Leaky ReLU')
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')
ax.spines['top'].set_position('zero')
ax.spines['right'].set_position('zero')
ax.set_ylim(-4,7)
# ax.set_xlabel('t', x=5)
# ax.set_ylabel('$\sigma(t)$', y=5)
ax.legend()
# fig.suptitle("Bijective Activation Functions")
fig.savefig(f"sigma.svg", format='svg')


fig, ax = plt.subplots()
ax.plot(y, x1, color = 'k', linestyle='solid', linewidth=2, label=f"Hyperbola (a=0.1, b=1, c=1, d=1)")
ax.plot(y, x2, color = 'k', linestyle='dotted', linewidth=2, label=f"Hyperbola (a=0.4, b=3, c=5, d=5)")
ax.plot(y, x3, color = 'k', linestyle='dashed', linewidth=2, label='Arcsinh')
ax.plot(y, x4, color = 'k', linestyle='dashdot', linewidth=2, label='Leaky ReLU')
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')
ax.spines['top'].set_position('zero')
ax.spines['right'].set_position('zero')
ax.set_ylim(-4,7)
# ax.set_xlabel('t', x=5)
# ax.set_ylabel('$\sigma(t)$', y=5)
ax.legend()
# fig.suptitle("Inverse Activation Functions")
fig.savefig(f"sigma-inv.svg", format='svg')