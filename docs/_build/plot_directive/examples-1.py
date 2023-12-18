import kerch

tr_set, _, _, _ = kerch.dataset.factory("two_moons",   # which dataset
                                     tr_size=250)      # training size
mdl = kerch.model.LSSVM(type="rbf",                    # kernel type
                     representation="dual")            # initiate model
mdl.set_data_prop(data=tr_set[0],                      # data
                  labels=tr_set[1],                    # corresponding labels
                  proportions=[1, 0, 0])               # initiate dataset
mdl.hyperopt({"gamma", "sigma"},                       # define which parameters to tune
             max_evals=500,                            # define how many trials
             k=10)                                     # 10-fold cross-validation
mdl.fit()                                              # fit the optimal parameters found
kerch.plot.plot_model(mdl)                             # plot the model using the built-in method