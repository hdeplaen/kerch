import kerpy as kp

## DATASET
tr_set, _, _, _ = kp.dataset.factory("two_moons")           # get the dataset
X, y = tr_set                                               # get data and labels

## MODEL & TRAINING
mdl = kp.model.LSSVM(type="rbf", representation="dual")     # initiate model
mdl.set_data_prop(X, y, proportions=[.8, .2, 0])            # initiate dataset
mdl.hyperopt({"sigma", "gamma"}, max_evals=1000)            # find optimal hyper-parameters
mdl.fit()                                                   # fit the optimal parameters found

## PLOT
kp.plot.plot_model(mdl)                                     # plot the model using the built-in method
