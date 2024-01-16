# coding=utf-8
import kerch
from sklearn.datasets import load_digits
from matplotlib import pyplot as plt
plt.axis('off')

COMPONENTS = 25
METHOD = 'iterative'
NUM = 10

# DATASET
NUM_TRAINING, NUM_TEST = 100, 50
digits = load_digits().data
training = digits[:NUM_TRAINING, :]
test = digits[NUM_TRAINING:NUM_TRAINING + NUM_TEST, :]

# MODEL
model = kerch.rkm.KPCA(sample=training, dim_output=COMPONENTS)
model.fit()

# KERNEL MATRIX
plt.figure()
fig, axs = plt.subplots(1,2)
axs[0].imshow(model.K)
axs[0].set_title('Original')
axs[0].axis('off')
axs[1].imshow(model.K_reconstructed)
axs[1].set_title('Learned')
axs[1].axis('off')
fig.suptitle('Kernel Matrix')
fig.show()

# RECONSRUCT
training_k = model.reconstruct()
training_recon = model.kernel.implicit_preimage(training_k, method=METHOD) #, num=NUM)
test_k = model.reconstruct(test)
test_recon = model.kernel.implicit_preimage(test_k, method=METHOD) #, num=NUM)

# VISUALIZE
TR_NUM = 25
TEST_NUM = 7

plt.figure()
plt.gray()
fig, axs = plt.subplots(2, 2)
axs[0, 0].imshow(training[TR_NUM, :].reshape(8, 8))
axs[0, 0].set_title("Training Original")
axs[0, 0].axis('off')
axs[0, 1].imshow(training_recon[TR_NUM, :].reshape(8, 8))
axs[0, 1].set_title("Training Recon.")
axs[0, 1].axis('off')
axs[1, 0].imshow(test[TEST_NUM, :].reshape(8, 8))
axs[1, 0].set_title("Test Original")
axs[1, 0].axis('off')
axs[1, 1].imshow(test_recon[TEST_NUM, :].reshape(8, 8))
axs[1, 1].set_title("Test Recon.")
axs[1, 1].axis('off')
fig.suptitle("Digits Comparison")
fig.show()
