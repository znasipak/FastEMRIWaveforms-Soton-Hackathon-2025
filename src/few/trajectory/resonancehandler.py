import numpy as np
from scipy.optimize import brentq

from ..utils.utility import (
    ELQ_to_pex,
    get_kerr_geo_constants_of_motion,
    get_fundamental_frequencies,
)

class ResonanceHandler:
    
    def __init__(self, res_list):

        self.res_list = res_list

        #Check if kappa_f or f_res are missing from the resonance description and if so set them to zero
        for res in self.res_list:
            if ('kappa_f' not in res) or ('f_res' not in res):
                res['kappa_f'] = 0
                res['f_res'] = lambda a, p, e, x: 0
    
        # Verbose debugging information
        # Set to 0 to turn off debugging output
        # Set to 1 to print verbose debugging output at the resonaces
        # Set to 2 to print verbose debugging at each time step
        self.verbose = 0

        # The first time we run this code we need to store the signs of the resonance conditions
        self.first_run = 1

    # This is the function called at each time step by the integrator action_function
    def check_for_resonance_crossing(self, t, y, spline_info, integrator):
        # Check: can the integrator variables be other than p,e,x?
        p, e, x = y[:3]

        # Exact the coefficients of the spline over the current time step
        rcont1 = spline_info[:,  0]    
        rcont2 = spline_info[:,  1]
        rcont3 = spline_info[:,  2]
        rcont4 = spline_info[:,  3]
        rcont5 = spline_info[:,  4]
        rcont6 = spline_info[:,  5]
        rcont7 = spline_info[:,  6]
        rcont8 = spline_info[:,  7]

        # function to calculate the spline at arbitrary values of s in [0,1]
        def y_of_s(s):
            s1 = 1.0 - s
            s2 = s**2
            s3 = s**3
            s4 = s**4
            s5 = s**5
            s6 = s**6
            return rcont1 + s * (rcont2 + s1 * (rcont3 + s
                            * (rcont4 + s1 * (rcont5 + s * (rcont6 + s1 * (rcont7 + s * rcont8))))))
        
        # function to calculate the derivatives of the phases w.r.t. s = (t - t0)/(Delta t)
        def dPhi_alpha_by_ds(s): 
            s2 = s**2
            s3 = s**3
            s4 = s**4
            s5 = s**5
            s6 = s**6
            
            return (
            rcont2[3:6]
            + rcont3[3:6] * (1 - 2 * s)
            + rcont4[3:6] * (2 * s - 3 * s2)
            + rcont5[3:6] * (2 * s - 6 * s2 + 4 * s3)
            + rcont6[3:6] * (3 * s2 - 8 * s3 + 5 * s4)
            + rcont7[3:6] * (3 * s2 - 12 * s3 + 15 * s4 - 6 * s5)
            + rcont8[3:6] * (4 * s3 - 15 * s4 + 18 * s5 - 7 * s6)
            )

        # To convert to t we need Delta t
        t_step_minus1 = integrator._integrator_t_cache[integrator.traj_step - 1]/integrator.Msec
        Deltat = t - t_step_minus1

        # Function to calculate the resonance conditions for every surface
        def surface_def(s):
            Omega_phi_spline, Omega_theta_spline, Omega_r_spline = dPhi_alpha_by_ds(s)/Deltat
            
            return list(map(lambda x: x['kappa_r']*Omega_r_spline + x['kappa_theta']*Omega_theta_spline + x['kappa_phi']*Omega_phi_spline + x['kappa_f']*x['f_res'](integrator.a,p,e,x), self.res_list))
        
        # The first time this function is run we need to store the values of the res. conds. at s=0
        if(self.first_run == 1):
            self.sign0 = np.sign(surface_def(0))
            self.first_run = 0
            if(self.verbose == 2): print("Initial signs of res cond:", self.sign0)

        if(self.verbose == 2): print(t, "res. condition: ", surface_def(1))

        # Calculate the res. conds at s=1
        self.sign1 = np.sign(surface_def(1))
        
        # Check if we cross a resonance by checking for a sign change in any of the resonance conditions
        if((self.sign1 != self.sign0).any()):
            if(self.verbose): 
                print("\nIntegrator crossed ", int(np.sum(np.absolute(self.sign0 - self.sign1))/2), " surface(s) on this step")
                print("At least one surface crossed near t = ", t, " where p = ", p, ", e = ", e, " x = ", x)

            # List of indices of surfaces crossed
            surfaces_crossed = np.nonzero(self.sign0 - self.sign1)[0]
            if(self.verbose): print("Resonance surfaces crossed: ", surfaces_crossed)
            
            # For every surface crossed find the value of s where the crossing happens
            surfaces_crossed_s_values = np.zeros(len(surfaces_crossed))
            for i, surface_index in enumerate(surfaces_crossed):
                surfaces_crossed_s_values[i] = brentq(lambda s: surface_def(s)[int(surface_index)], 0, 1)

            # We want to stop at the first surface crossed, which will have the smallest s value
            min_s_value = np.min(surfaces_crossed_s_values)
            min_s_value_index = np.argmin(surfaces_crossed_s_values)
            
            # Calculate the time and (p,e,x) when the first surface is crossed
            s_surface = min_s_value
            t_surface = s_surface*Deltat + t_step_minus1
            p_surface, e_surface, x_surface, Phi_phi_surface, Phi_theta_surface, Phi_r_surface = y_of_s(s_surface)
            if(self.verbose): print("Surface at s = ", s_surface)
            if(self.verbose): print("Surface at t = ", t_surface, " where p = ", p_surface, ", e = ", e_surface, " x = ", x_surface)

            if(self.verbose): print("Res. cond. on first surface:", surface_def(s_surface))

            # Calculate the jumps on the first surface crossed
            E_surface, L_surface, Q_surface = get_kerr_geo_constants_of_motion(integrator.a, p_surface, e_surface, x_surface)
            jump_E, jump_L, jump_Q = self.res_list[min_s_value_index]['jump_func'](integrator.a, e_surface, x_surface)
            new_p, new_e, new_x = ELQ_to_pex(integrator.a, E_surface + jump_E, L_surface + jump_L, Q_surface + jump_Q)

            # What do we do if applying the jumps pushes the trajectory across another surface?
            # At the moment this causes the code to crash.
        
            t = t_surface
            y[0] = new_p
            y[1] = new_e
            y[2] = new_x
            y[3] = Phi_phi_surface
            y[4] = Phi_theta_surface
            y[5] = Phi_r_surface
            
            if(self.verbose): print("Parameters after resonances = ", t_surface, " where p = ", y[0], ", e = ", y[1], " x = ", y[2], "\n")
            
            #update the spline info (computed in by Niels in the Mathematica FEW_splines.nb)
            spline_info[:, 0] = rcont1
            spline_info[:, 1] = s_surface*(rcont2 - (-1 + s_surface)*(rcont3 + s_surface*(rcont4 - (-1 + s_surface)*(rcont5 + s_surface*(rcont6 - (-1 + s_surface)*(rcont7 + rcont8*s_surface))))))
            spline_info[:, 2] = s_surface**2*(rcont3 + (-1 + s_surface)*(rcont4 - (-1 + s_surface)*(rcont5 + s_surface*(rcont6 - (-1 + s_surface)*(rcont7 + rcont8*s_surface)))))
            spline_info[:, 3] = s_surface**3*(rcont4 + (-1 + s_surface)*(-2*rcont5 + rcont6 - 3*rcont6*s_surface + (-1 + s_surface)*(rcont7*(-1 + 4*s_surface) + rcont8*s_surface*(-2 + 5*s_surface))))
            spline_info[:, 4] = s_surface**4*(rcont5 - (-1 + s_surface)*(-2*rcont6 + (-1 + s_surface)*(3*rcont7 - rcont8 + 4*rcont8*s_surface)))
            spline_info[:, 5] = s_surface**5*(rcont6 - 3*(-1 + s_surface)*(rcont7 + rcont8*(-1 + 2*s_surface)))
            spline_info[:, 6] = (rcont7 + 3*rcont8*(-1 + s_surface))*s_surface**6
            spline_info[:, 7] = rcont8*s_surface**7

            # update sign0 to show we have crossed a resonances surface
            self.sign0[surfaces_crossed[min_s_value_index]] = self.sign1[surfaces_crossed[min_s_value_index]]
            
        # we need to return the t and y on the resonance surface, and also the updated spline information (TODO)
        return t, y, spline_info
    
    