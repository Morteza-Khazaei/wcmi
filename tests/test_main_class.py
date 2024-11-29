from wcmi import VegParamCal



opt_obj = VegParamCal(dir_radar_sigma='datasets/backscatter', 
                      dir_risma='datasets/RISMA', 
                      aafc_croptype=[158, ], 
                      risma_station=['MB1', 'MB5'], 
                      sensor_depth=[0, 5])
wcm_param = opt_obj.run()
print(wcm_param)