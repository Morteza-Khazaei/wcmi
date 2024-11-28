from wcmi import VegParamCal



opt_obj = VegParamCal(dir_radar_sigma='datasets/backscatter', 
                      dir_risma='datasets/RISMA')
opt_obj.run()