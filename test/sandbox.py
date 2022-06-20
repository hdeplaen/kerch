import kerch

mv = kerch.rkm.MultiView({"type": "rbf", "sample": range(10)},
                         {"type": "linear", "sample": range(10)},
                         dim_output=3)
print(mv.K)
