import numpy as np
import matplotlib.pyplot as plt
from vdw import vdWclass

Nat = 60
model = 'MBD'

from inout import read_optsteps, plot_single_polmat, plot_3Dstruct
from inout import plot_1Dproj

steps = read_optsteps('steps' + str(model) + '.txt', Nat)
md = np.loadtxt('MD/structMBD-MBD.txt')

vdw = vdWclass()

from inout import plot_polmat_group, plot_polar_group, plot_opt_evol

## polar coordinate analysis
plot_polar_group(steps)

## total polarisability evolution
plot_opt_evol(steps, vdw)

#### heron only for MBD

## polmats and structs
plot_3Dstruct(steps[0], out = 'fig_struct_input.pdf', view_ang=[21, -79])
plot_3Dstruct(steps[-1], out = 'fig_struct_opt.pdf', view_ang=[21, -79])
plot_3Dstruct(md, out = 'fig_struct_md.pdf', view_ang=[21, -79])

plot_single_polmat(steps[0], vdw, label = 'input', out = 'fig_polmat_input.pdf')
plot_single_polmat(steps[-1], vdw, label = 'optimised', out = 'fig_polmat_opt.pdf')
plot_single_polmat(md, vdw, label = 'MD', out = 'fig_polmat_md.pdf')
    
## 1D projections of the polarisability
plot_1Dproj(steps[0], vdw, label='input', out='fig_1Dproj_input.pdf')
plot_1Dproj(steps[-1], vdw, label='optimised', out='fig_1Dproj_opt.pdf')
plot_1Dproj(md, vdw, label='MD', out='fig_1Dproj_md.pdf')

# histogram
from inout import plot_1Dproj_hist
plot_1Dproj_hist(steps[0], vdw, label = 'input', out='fig_1DprojHist_input.pdf')
plot_1Dproj_hist(steps[-1], vdw, label = 'opt', out='fig_1DprojHist_opt.pdf')
plot_1Dproj_hist(md, vdw, label = 'MD', out='fig_1DprojHist_md.pdf')
