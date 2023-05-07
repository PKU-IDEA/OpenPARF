##
# @file   dct_unitest.py
# @author Yibo Lin
# @date   Mar 2019
#

import pdb
import os
import sys
import numpy as np
import unittest

import torch
from torch.autograd import Function, Variable
import time
import scipy
from scipy import fftpack

if len(sys.argv) < 2:
    print("usage: python script.py [project_dir]")
    project_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
else:
    project_dir = sys.argv[1]
print("use project_dir = %s" % (project_dir))

sys.path.append(project_dir)
from openparf.ops.dct import dct, discrete_spectral_transform
import openparf.configure as configure
sys.path.pop()

dtype = torch.float32


class DCTOpTest(unittest.TestCase):
    def testDct2Random(self):
        torch.manual_seed(10)
        M = 4
        N = 8
        x = torch.empty(M, N, dtype=dtype).uniform_(0, 10.0)
        expkM = discrete_spectral_transform.getExactExpk(M,
                                                         dtype=x.dtype,
                                                         device=x.device)
        expkN = discrete_spectral_transform.getExactExpk(N,
                                                         dtype=x.dtype,
                                                         device=x.device)

        golden_value = discrete_spectral_transform.dct2_N(x).data.numpy()
        print("2D DCT golden_value")
        print(golden_value)

        # test cpu using fft2
        custom = dct.Dct2(expkM, expkN)
        dct_value = custom.forward(x)
        print("2D dct_value")
        print(dct_value.data.numpy())

        np.testing.assert_allclose(dct_value.data.numpy(),
                                   golden_value,
                                   rtol=1e-6,
                                   atol=1e-5)

        if torch.cuda.device_count():
            # test gpu using fft2
            custom = dct.Dct2(expkM.cuda(), expkN.cuda())
            dct_value = custom.forward(x.cuda()).cpu()
            print("2D dct_value cuda")
            print(dct_value.data.numpy())

            np.testing.assert_allclose(dct_value.data.numpy(),
                                       golden_value,
                                       rtol=1e-6,
                                       atol=1e-5)

    def testIdct2Random(self):
        torch.manual_seed(10)
        M = 4
        N = 8
        x = torch.empty(M, N, dtype=torch.int32).random_(0, 10).double()
        print("2D x")
        print(x)

        expkM = discrete_spectral_transform.getExactExpk(M,
                                                         dtype=x.dtype,
                                                         device=x.device)
        expkN = discrete_spectral_transform.getExactExpk(N,
                                                         dtype=x.dtype,
                                                         device=x.device)

        y = discrete_spectral_transform.dct2_2N(x)

        golden_value = discrete_spectral_transform.idct2_2N(y).data.numpy()
        print("2D idct golden_value")
        print(golden_value)

        golden_scipy_value = fftpack.idct(fftpack.idct(y.data.numpy()).T).T
        print("2D idct golden_scipy_value")
        print(golden_scipy_value)

        np.testing.assert_allclose(golden_scipy_value,
                                   golden_value,
                                   rtol=1e-6,
                                   atol=1e-5)

        # test cpu using fft2
        custom = dct.Idct2(expkM, expkN)
        dct_value = custom.forward(y)
        print("2D idct_value cuda")
        print(dct_value.data.numpy())

        np.testing.assert_allclose(dct_value.data.numpy(),
                                   golden_value,
                                   rtol=1e-6,
                                   atol=1e-5)

        if torch.cuda.device_count():
            # test gpu using ifft2
            custom = dct.Idct2(expkM.cuda(), expkN.cuda())
            dct_value = custom.forward(y.cuda()).cpu()
            print("2D idct_value cuda")
            print(dct_value.data.numpy())

            np.testing.assert_allclose(dct_value.data.numpy(),
                                       golden_value,
                                       rtol=1e-6,
                                       atol=1e-5)


class DXTOpTest(unittest.TestCase):
    def testIdctIdxstRandom(self):
        torch.manual_seed(10)
        M = 4
        N = 8
        x = torch.empty(M, N, dtype=torch.int32).random_(0, 10).double()
        print("2D x")
        print(x)

        expkM = discrete_spectral_transform.getExactExpk(M,
                                                         dtype=x.dtype,
                                                         device=x.device)
        expkN = discrete_spectral_transform.getExactExpk(N,
                                                         dtype=x.dtype,
                                                         device=x.device)

        golden_value = discrete_spectral_transform.idctIdxst(x).data.numpy()
        print("2D golden_value")
        print(golden_value)

        # test gpu
        custom = dct.IdctIdxst(expkM, expkN)
        idct_idxst_value = custom.forward(x)
        print("2D dct2_fft2.idctIdxst cuda")
        print(idct_idxst_value.data.numpy())

        # note the scale factor
        np.testing.assert_allclose(idct_idxst_value.data.numpy(),
                                   golden_value * 2,
                                   atol=1e-14)

        if torch.cuda.device_count():
            # test gpu
            custom = dct.IdctIdxst(expkM.cuda(), expkN.cuda())
            idct_idxst_value = custom.forward(x.cuda()).cpu()
            print("2D dct2_fft2.idctIdxst cuda")
            print(idct_idxst_value.data.numpy())

            # note the scale factor
            np.testing.assert_allclose(idct_idxst_value.data.numpy(),
                                       golden_value * 2,
                                       atol=1e-14)

    def testIdxstIdctRandom(self):
        torch.manual_seed(10)
        M = 4
        N = 8
        x = torch.empty(M, N, dtype=torch.int32).random_(0, 10).double()
        print("2D x")
        print(x)

        expkM = discrete_spectral_transform.getExactExpk(M,
                                                         dtype=x.dtype,
                                                         device=x.device)
        expkN = discrete_spectral_transform.getExactExpk(N,
                                                         dtype=x.dtype,
                                                         device=x.device)

        golden_value = discrete_spectral_transform.idxstIdct(x).data.numpy()
        print("2D golden_value")
        print(golden_value)

        # test cpu
        custom = dct.IdxstIdct(expkM, expkN)
        idxst_idct_value = custom.forward(x)
        print("2D dct2_fft2.idxstIdct cuda")
        print(idxst_idct_value.data.numpy())

        # note the scale factor
        np.testing.assert_allclose(idxst_idct_value.data.numpy(),
                                   golden_value * 2,
                                   atol=1e-14)

        if torch.cuda.device_count():
            # test gpu
            custom = dct.IdxstIdct(expkM.cuda(), expkN.cuda())
            idxst_idct_value = custom.forward(x.cuda()).cpu()
            print("2D dct2_fft2.idxstIdct cuda")
            print(idxst_idct_value.data.numpy())

            # note the scale factor
            np.testing.assert_allclose(idxst_idct_value.data.numpy(),
                                       golden_value * 2,
                                       atol=1e-14)


if __name__ == '__main__':
    torch.manual_seed(10)
    np.random.seed(10)

    if len(sys.argv) < 2:
        pass
    else:
        sys.argv.pop()
    unittest.main()
