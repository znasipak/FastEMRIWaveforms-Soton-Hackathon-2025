import few
from few.trajectory.ode.base import ODEBase
from few.trajectory.inspiral import EMRIInspiral
from few.trajectory.ode import SchwarzEccFlux, SchwarzCirc0PAScalar
from few.utils.utility import get_fundamental_frequencies, get_separatrix, ELQ_to_pex
from few.waveform import FastSchwarzschildCircularFluxScalar, FastSchwarzschildEccentricFlux, GenerateEMRIWaveform

from few.waveform.base import SphericalHarmonicWaveformBase
from few.amplitude.romannet import RomanAmplitude
from few.utils.modeselector import ModeSelector, NeuralModeSelector
from few.summation.interpolatedmodesum import InterpolatedModeSum
from few.summation.fdinterp import FDInterpolatedModeSum

from few.utils.baseclasses import (
    SchwarzschildEccentric,
    KerrEccentricEquatorial,
    Pn5AAK,
    ParallelModuleBase,
)


from few.mappings import ELdot_to_PEdot_Jacobian

import numpy as np 
from typing import Union, Optional
import matplotlib.pyplot as plt 
from multispline.spline import BicubicSpline, TricubicSpline, CubicSpline
from math import pow, sqrt, log

#from .utils.globals import get_logger, get_config, get_file_manager

# define trajectory RHS class
class SchwarzCirc1PA(ODEBase):
    """
    Schwarzschild circular flux ODE.
    Args:
        use_ELQ: If True, the ODE will output derivatives of the orbital elements of (E, L, Q). Defaults to False.
    """

    def __init__(self, *args, use_ELQ: bool = False, **kwargs):
        super().__init__(*args, use_ELQ=use_ELQ, **kwargs)

        data = np.loadtxt("/home/people/sbarsanti/FEW/pip_few/kerr/FastEMRIWaveforms/QuasiCircularData_new.dat")
        
        x = np.unique(data[:, 0])

        self.pdot_interp_0pa =  CubicSpline(x, data[:, 1])
        self.pdot_interp_1pa =  CubicSpline(x, data[:, 2])

    @property
    def equatorial(self):
        return True

    @property
    def background(self):
        return "Schwarzschild"

    @property
    def separatrix_buffer_dist(self):
        return 0.1

    @property
    def supports_ELQ(self):
        return False

    @property
    def flux_output_convention(self):
        return "pex"

    def distance_to_outer_boundary(self, y):
        p, e, x = self.get_pex(y)
        dist_p = 3.817 - np.log((p - 2. * e - 2.1))
        dist_e = 0.75 - e

        if dist_p < 0 or dist_e < 0:
            mult = -1
        else:
            mult = 1

        dist = mult * min(abs(dist_p), abs(dist_e))
        return dist

    def interpolate_flux_grids(
              self, p: float
              ) -> tuple[float]:

        if p < 6.25 or p > 30.: 
            raise ValueError("Interpolation: p out of bounds.")

        f0PA0PN = - 64./(5*(p**3))
        f1PA0PN = 176./(5*(p**4))

        pdot_0pa = (self.pdot_interp_0pa(p)/(p**4))+f0PA0PN
        if self.additional_args[0] != 1 and self.additional_args[0] != 0 :  #setting 1pa or 0pa
            raise ValueError("args for 1PA must be either 0 (for 0PA) or 1 (for 1PA)")
        pdot_1pa = self.additional_args[0]*(((self.pdot_interp_1pa(p))/(p**5))+f1PA0PN)
        return [pdot_0pa, pdot_1pa]

    def evaluate_rhs(
        self, y: Union[list[float], np.ndarray]
    ) -> list[Union[float, np.ndarray]]:
        if self.use_ELQ:
            E, L, Q = y[:3]
            p, e, x = ELQ_to_pex(self.a, E, L, Q)

        else:
            p, e, x = y[:3]
        #eps=mu/M
        eps = self.epsilon
        nu = eps/((1+eps)**2)
        Omega_phi, Omega_theta, Omega_r = get_fundamental_frequencies(self.a, p, e, x)
        pdot_0pa, pdot_1pa = self.interpolate_flux_grids(p)
         
        pdot  = (nu/eps)*(pdot_0pa+(nu*pdot_1pa))
        return [pdot, 0., 0., Omega_phi, Omega_theta, Omega_r]


class FastSchwarzschildCircular1PAFlux(
    SphericalHarmonicWaveformBase, SchwarzschildEccentric):
    """Prebuilt model for fast Schwarzschild eccentric flux-based waveforms.

    This model combines the most efficient modules to produce the fastest
    accurate EMRI waveforms. It leverages GPU hardware for maximal acceleration,
    but is also available on for CPUs.

    The trajectory module used here is :class:`few.trajectory.inspiral` for a
    flux-based, sparse trajectory. This returns approximately 100 points.

    The amplitudes are then determined with
    :class:`few.amplitude.romannet` along these sparse
    trajectories. This gives complex amplitudes for all modes in this model at
    each point in the trajectory. These are then filtered with
    :class:`few.utils.modeselector.ModeSelector`.

    The modes that make it through the filter are then summed by
    :class:`few.summation.interpolatedmodesum.InterpolatedModeSum`.

    See :class:`few.waveform.base.SphericalHarmonicWaveformBase` for information
    on inputs. See examples as well.

    args:
        inspiral_kwargs: Optional kwargs to pass to the
            inspiral generator. **Important Note**: These kwargs are passed
            online, not during instantiation like other kwargs here. Default is
            {}.
        amplitude_kwargs: Optional kwargs to pass to the
            amplitude generator during instantiation. Default is {}.
        sum_kwargs: Optional kwargs to pass to the
            sum module during instantiation. Default is {}.
        Ylm_kwargs: Optional kwargs to pass to the
            Ylm generator during instantiation. Default is {}.
        use_gpu: If True, use GPU resources. Default is False.
        *args: args for waveform model.
        **kwargs: kwargs for waveform model.

    """
    def __init__(
        self,
        inspiral_kwargs: Optional[dict]=None,
        amplitude_kwargs: Optional[dict]=None,
        sum_kwargs: Optional[dict]=None,
        Ylm_kwargs: Optional[dict]=None,
        mode_selector_kwargs: Optional[dict]=None,
        use_gpu: bool=False,
        *args: Optional[tuple],
        **kwargs: Optional[dict],
    ):
        if inspiral_kwargs is None:
            inspiral_kwargs = {}
        if amplitude_kwargs is None:
            amplitude_kwargs = {}
        if sum_kwargs is None:
            sum_kwargs = {}
        if Ylm_kwargs is None:
            Ylm_kwargs = {}
        if mode_selector_kwargs is None:
            mode_selector_kwargs = {}

        SchwarzschildEccentric.__init__(self, use_gpu=use_gpu, nmax=30)

        inspiral_kwargs["func"] = SchwarzCirc1PA
        # inspiral_kwargs = augment_ODE_func_name(inspiral_kwargs)

        mode_summation_module = InterpolatedModeSum
        if "output_type" in sum_kwargs:
            if sum_kwargs["output_type"] == "fd":
                mode_summation_module = FDInterpolatedModeSum

        mode_selection_module = ModeSelector
        if "mode_selection_type" in mode_selector_kwargs:
            if mode_selector_kwargs["mode_selection_type"] == "neural":
                mode_selection_module = NeuralModeSelector
                if "mode_selector_location" not in mode_selector_kwargs:
                    mode_selector_kwargs["mode_selector_location"] = os.path.join(
                        dir_path,
                        "./files/modeselector_files/KerrEccentricEquatorialFlux/",
                    )
                mode_selector_kwargs["keep_inds"] = np.array([0, 1, 3, 4, 6, 7, 8, 9])

        SphericalHarmonicWaveformBase.__init__(
            self,
            EMRIInspiral,
            RomanAmplitude,
            mode_summation_module,
            mode_selection_module,
            inspiral_kwargs=inspiral_kwargs,
            amplitude_kwargs=amplitude_kwargs,
            sum_kwargs=sum_kwargs,
            Ylm_kwargs=Ylm_kwargs,
            mode_selector_kwargs=mode_selector_kwargs,
            use_gpu=use_gpu,
            normalize_amps=True,
            *args,
            **kwargs,
        )

    @property
    def gpu_capability(self):
        return True

    @property
    def allow_batching(self):
        return False

    def __call__(
        self,
        M: float,
        mu: float,
        p0: float,
        e0: float,
        theta: float,
        phi: float,
        *args: Optional[tuple],
        **kwargs: Optional[dict],
    ) -> np.ndarray:
        """
        Generate the waveform.

        Args:
            M: Mass of larger black hole in solar masses.
            mu: Mass of compact object in solar masses.
            p0: Initial semilatus rectum of inspiral trajectory.
            e0: Initial eccentricity of inspiral trajectory.
            theta: Polar angle of observer.
            phi: Azimuthal angle of observer.
            *args: Placeholder for additional arguments.
            **kwargs: Placeholder for additional keyword arguments.

        Returns:
            Complex array containing generated waveform.

        """
        # insert missing arguments for this waveform class
        return self._generate_waveform(
            M,
            mu,
            0.0,
            p0,
            e0,
            1.0,
            theta,
            phi,
            *args,
            **kwargs,
        )


try:
    import cupy as cp
    use_gpu = True
    xp = cp
except ImportError:
    use_gpu = False


M = 1e6
mu = 1e1
p0 = 9.48
e0 = 0.
x0 = 1. 
a0 = 0. 
T = 1.
dt = 10. 
dist = 1. 

print("trajectory")
traj = EMRIInspiral(func=SchwarzEccFlux,use_gpu=use_gpu)
test = traj(M, mu, 0.0, p0, e0, 1.0, T=T, dt=10.0)
t = test[0]
p = test[1]
print("GR_computed")
trajspin = EMRIInspiral(func=SchwarzCirc1PA,use_gpu=use_gpu)
PAorder = 1. 
onepaspin = trajspin(M, mu, 0.0, p0, e0, 1.0, PAorder, T=T, dt=10.0)
print("1PA_spin_computed")

plt.plot(onepaspin[0], onepaspin[1], label='1PA ')
plt.plot(t, p, label='0PA - true')
plt.ylabel('p')
plt.xlabel('t')

plt.legend()
plt.savefig("fig_1pa_spin_trajectory.pdf")


# keyword arguments for inspiral generator (RunSchwarzEccFluxInspiral)
inspiral_kwargs={
        "DENSE_STEPPING": 0,  # we want a sparsely sampled trajectory
        "max_init_len": int(1e7),  # all of the trajectories will be well under len = 1e7 (default 1e3)
        "err": 1e-12,  # To be set within the class  #in the example is 1e-10. it was working with 5e-12. now i try 1e-11.
        "use_rk4": True,
 }

# keyword arguments for inspiral generator (RomanAmplitude)
amplitude_kwargs = {
    "max_init_len": int(1e6),  # all of the trajectories will be well under len = 1000
    "use_gpu": use_gpu  # GPU is available in this class
}

# keyword arguments for Ylm generator (GetYlms)
Ylm_kwargs = {
    "assume_positive_m": False  # if we assume positive m, it will generate negative m for all m>0
}

# keyword arguments for summation generator (InterpolatedModeSum)
sum_kwargs = {
    "use_gpu": use_gpu,  # GPU is availabel for this type of summation
    "pad_output": False,  #True? 
}

qS = 0.2
phiS = 0.2
qK = 0.8
phiK = 0.8
Phi_phi0 = 0.
Phi_theta0 = 0.
Phi_r0 = 0.
a=0.

funcwave = 'FastSchwarzschildCircular1PASpinFlux'

wave = FastSchwarzschildCircular1PAFlux()
 
PAorder = 1. 
       
h = wave(M, mu, p0, e0, qS, phiS,  PAorder)
waveform = xp.asarray(h)

print("waveform1PA=", waveform)

#to do: GenerateEMRIWaveform
