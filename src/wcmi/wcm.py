import numpy as np

def WCM(A, B, V1, V2, theta_rad, sigma_soil):
    # Water Cloud Model (WCM)
    
    cos_theta = np.cos(theta_rad)
    
    tau = np.exp(-2 * B * V2 / cos_theta)
    
    sigma_veg = A * V1 * cos_theta * (1 - tau)
    
    sigma_tot = sigma_veg + (tau * sigma_soil)

    return sigma_tot, sigma_veg, tau