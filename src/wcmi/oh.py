import numpy as np
import matplotlib.pyplot as plt




"""
Major surface scatter class
"""
class SurfaceScatter(object):
    def __init__(self, eps=None, ks=None, theta=None, kl=None, mv=None, C_hh=None, C_vv=None, C_hv=None, D_hh=None, D_vv=None, D_hv=None, **kwargs):
        self.eps = eps
        self.ks = ks
        self.theta = theta
        self.kl = kl

        self.mv = mv
        self.C_hh = C_hh
        self.C_vv = C_vv
        self.D_hh = D_hh
        self.D_vv = D_vv
        self.C_hv = C_hv
        self.D_hv = D_hv

        self._check()

    def _check(self):
        pass
        # assert isinstance(self.eps, complex)


class Oh04(SurfaceScatter):
    def __init__(self, mv, ks, theta):
        """
        Parameters
        ----------
        mv : float, ndarray
            volumetric soil moisture m3/m3
        ks : float
            product of wavenumber and rms height
            be aware that both need to have the same units
        theta : float, ndarray
            incidence angle [rad]
        """
        super(Oh04, self).__init__(mv=mv, ks=ks, theta=theta)

        # calculate p and q
        self._calc_p()
        self._calc_q()

        # calculate backascatter
        self.vh = self._calc_vh()
        # difference between hv and vh?
        self.vv = self.vh / self.q
        self.hh = self.vh / self.q * self.p

    def get_sim(self):
        return self.vh, self.vv, self.hh

    def _to_dB(self, linear):
        return 10*np.log10(linear)

    def _calc_p(self):
        self.p = 1 - (2.*self.theta/np.pi)**(0.35*self.mv**(-0.65)) * np.exp(-0.4 * self.ks**1.4)

    def _calc_q(self):
        self.q = 0.095 * (0.13 + np.sin(1.5*self.theta))**1.4 * (1-np.exp(-1.3 * self.ks**0.9))

    def _calc_vh(self):
        a = 0.11 * self.mv**0.7 * np.cos(self.theta)**2.2
        b = 1 - np.exp(-0.32 * self.ks**1.8)
        return a*b

    def plot(self, ptype='mv'):
        f = plt.figure()
        ax = f.add_subplot(111)

        if ptype == 'mv':
          x = self.mv
          ax.set_xlabel('volumetric soil moisture [m^3/m^3]')
        elif ptype == 'ks':
          x = self.ks
          ax.set_xlabel('rms height [m]')
        elif ptype == 'theta':
          x = np.rad2deg(self.theta)
          ax.set_xlabel('incidence angle [deg]')
        else:
          raise ValueError('Unknown parameter type')

        ax.plot(x, 10.*np.log10(self.hh), color='blue', label='hh')
        ax.plot(x, 10.*np.log10(self.vv), color='red', label='vv')
        ax.plot(x, 10.*np.log10(self.vh), color='green', label='vh')
        ax.grid()
        #ax.set_ylim(-25.,0.)
        #ax.set_xlim(0.,70.)
        ax.legend()
        # ax.set_xlabel('incidence angle [deg]')
        ax.set_ylabel('backscatter [dB]')