import kerch

## DATASET
tr_set, _, _, _ = kerch.dataset.factory("two_moons",
                                        tr_size=250)           # get the dataset
X, y = tr_set                                               # get data and labels

## MODEL & TRAINING
mdl = kerch.model.LSSVM(type="rbf", representation="dual")     # initiate model
mdl.set_data_prop(X, y, proportions=[1, 0, 0])              # initiate dataset
mdl.fit()
kerch.plot.plot_model(mdl)                                     # plot the model using the built-in method

## TUNING
mdl.hyperopt({"gamma", "sigma"}, max_evals=50, k=10)       # find optimal hyper-parameters
mdl.fit()                                                   # fit the optimal parameters found
kerch.plot.plot_model(mdl)                                     # plot the model using the built-in method
