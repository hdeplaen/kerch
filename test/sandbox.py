import kerch

mv = kerch.rkm.MultiView({"type": "rbf", "sample": range(10), 'name': 'space'},
                         {"type": "linear", "sample": range(10), 'name': 'time'},
                         dim_output=3)
print(mv.K)
