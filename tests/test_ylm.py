# python -m unittest few/tests/test_ylm.py
import numpy as np
import unittest

from few.utils.globals import get_logger, get_first_backend
from few.utils.ylm import GetYlms

few_logger = get_logger()

best_backend = get_first_backend(GetYlms.supported_backends())
few_logger.warning("Physics Test is running with backend {}".format(best_backend.name))


N_TESTS = 10


class ModuleTest(unittest.TestCase):
    def setUp(self):
        self.ylm = GetYlms(assume_positive_m=False, force_backend=best_backend)
        self.ylm_assume_positive_m = GetYlms(assume_positive_m=True, force_backend=best_backend)

    def tearDown(self):
        del self.ylm
        del self.ylm_assume_positive_m

    def test_pos_neg_m_index_symmetry(self):
        l_range = [2, 11]

        test_theta = np.pi / 3.0
        test_phi = np.pi / 4.0

        # check index symmetries ranging over positive m and positive n
        for l in range(l_range[0], l_range[1]):
            for m in range(1, l+1):
                assert (
                    (self.ylm(l, m, test_theta, test_phi) - (-1)**(l) * self.ylm(l, -m, test_theta, test_phi)) < 1e-14
                ), "Ylms not obeying m -> -m symmetry relation."

    def test_antipodal_symmetry(self):
        l_range = [2, 11]

        test_theta = np.pi / 3.0
        test_phi = np.pi / 4.0

        theta_antipode = np.pi - test_theta
        phi_antipode = test_phi + np.pi

        # check index symmetries ranging over positive m and positive n
        for l in range(l_range[0], l_range[1]):
            for m in range(1, l+1):
                assert (
                    (self.ylm(l, m, test_theta, test_phi) - (-1)**(l+m) * self.ylm(l, m, theta_antipode, phi_antipode)) < 1e-14
                ), "Ylms not obeying m -> -m symmetry relation."
