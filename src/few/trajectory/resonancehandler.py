import numpy as np
from scipy.optimize import brentq

from ..utils.utility import (
    get_fundamental_frequencies,
)

class ResonanceHandler:
    
    
    def __init__(self):
        self.after_res = 0
        
    # we need to return the t and y on the resonance surface, and also the updated spline information (TODO)    
    def check_for_resonance_crossing(self, t, y, spline_info, integrator):
        p, e, x = y[:3]
        
        rcont1 = spline_info[:,  0]    
        rcont2 = spline_info[:,  1]
        rcont3 = spline_info[:,  2]
        rcont4 = spline_info[:,  3]
        rcont5 = spline_info[:,  4]
        rcont6 = spline_info[:,  5]
        rcont7 = spline_info[:,  6]
        rcont8 = spline_info[:,  7]
        
        def y_of_s(s):
            s1 = 1.0 - s
            s2 = s**2
            s3 = s**3
            s4 = s**4
            s5 = s**5
            s6 = s**6
            return rcont1 + s * (
                        rcont2
                        + s1
                        * (
                            rcont3
                            + s
                            * (rcont4 + s1 * (rcont5 + s * (rcont6 + s1 * (rcont7 + s * rcont8))))
                        )
                    )
        
        # function to calculate the derivatives of the phases w.r.t. s = (t - t0)/(Delta t)
        def dPhi_alpha_by_ds(s): 
            s2 = s**2
            s3 = s**3
            s4 = s**4
            s5 = s**5
            s6 = s**6
            
            return (
            rcont2[4:6]
            + rcont3[4:6] * (1 - 2 * s)
            + rcont4[4:6] * (2 * s - 3 * s2)
            + rcont5[4:6] * (2 * s - 6 * s2 + 4 * s3)
            + rcont6[4:6] * (3 * s2 - 8 * s3 + 5 * s4)
            + rcont7[4:6] * (3 * s2 - 12 * s3 + 15 * s4 - 6 * s5)
            + rcont8[4:6] * (4 * s3 - 15 * s4 + 18 * s5 - 7 * s6)
            )
        
        Omega_phi_direct, Omega_theta_direct, Omega_r_direct = get_fundamental_frequencies(integrator.a,p,e,x)    
        
        # Evaluate the frequencies at the end of the ODE step
        Omega_theta_spline, Omega_r_spline = dPhi_alpha_by_ds(1)
        
        print(Omega_theta_spline/Omega_r_spline, Omega_theta_direct/Omega_r_direct, 2*Omega_theta_spline - 3*Omega_r_spline)
        
        def surface_def(s):
            Omega_theta_spline, Omega_r_spline = dPhi_alpha_by_ds(s)
            return 2*Omega_theta_spline - 3*Omega_r_spline
            
        # To convert to t we need Delta t
        t_step_minus1 = integrator._integrator_t_cache[integrator.traj_step - 1]/integrator.Msec
        Deltat = t - t_step_minus1
        
        # Check if we cross a resonance
        if((2*Omega_theta_spline - 3*Omega_r_spline > 0) and self.after_res == 0):
            print("Surface crossed near t = ", t, " where p = ", p, ", e = ", e, " x = ", x)
            s_surface = brentq(surface_def, 0, 1)
            t_surface = s_surface*Deltat + t_step_minus1
            p_surface, e_surface, x_surface, Phi_phi_surface, Phi_theta_surface, Phi_r_surface = y_of_s(s_surface)
            print("Surface at s = ", s_surface)
            print("Surface at t = ", t_surface, " where p = ", p_surface, ", e = ", e_surface, " x = ", x_surface)
            
            
            delta_p = 0;
            delta_e = 0.1;
            delta_x = 0;
        
            t = t_surface
            y[0] = p_surface + delta_p
            y[1] = e_surface + delta_e
            y[2] = x_surface + delta_x
            y[3] = Phi_phi_surface
            y[4] = Phi_theta_surface
            y[5] = Phi_r_surface
            
            print("Parameters after resonances = ", t_surface, " where p = ", y[0], ", e = ", y[1], " x = ", y[2])
            
        
            self.after_res = 1

        # we need to return the t and y on the resonance surface, and also the updated spline information (TODO)
        return t, y, spline_info
    
    