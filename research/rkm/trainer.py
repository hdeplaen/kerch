import kerch
import torch

learning_set = kerch.data.factory(dataset="pima", num_training=200, num_validation=0, num_test=50)

rkm = kerch.rkm.RKM(sample_projections=['center', 'normalize'])
rkm.append_level(level_type='kpca', constraint='soft', representation='dual', dim_output=5, sigma_trainble=True)
rkm.append_level(level_type='kpca', constraint='soft', representation='dual', dim_output=2, sigma_trainble=True)
rkm.append_level(level_type='lssvm', constraint='soft', representation='dual', dim_output=1, sigma_trainble=True)


trainer = kerch.rkm.Trainer(model=rkm, learning_set=learning_set, epochs=1000)
trainer.fit()