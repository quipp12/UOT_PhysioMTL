import pandas as pd
import numpy as np
from algo import revised_Hungarian
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import matplotlib
import time
from scipy.optimize import linear_sum_assignment
import networkx as nx
import math
from UOT_algo import sinkhorn_uot
import ot

##############################
# load and preprocess the data#
##############################
# %%

Data = pd.read_csv("/Users/quipham/Downloads/wdbc.data", header=None)

np.random.seed(1)
print(Data)
Xvar = Data[Data[1] == 'B'].iloc[:, 2:33]
Xvar = (Xvar / Xvar.max()).to_numpy()  # Normalization of pixel values (to [0-1] range)
Yvar = Data[Data[1] == 'M'].iloc[:, 2:33]
Yvar = (Yvar / Yvar.max()).to_numpy()

print(np.shape(Data))
########################################################################
# Compare between modified Hungarian and network simplex on breast data#
########################################################################
# %%intialize parameters before iterations (independent case
m = np.arange(5, 40, 5)
sample = np.tile(np.repeat(m, 10), 2)
l = np.repeat(np.array([1, 2]), 70)
settings = list(zip(sample, l))

# %%create list to store the results
i = 1
results_UOT_time_breast = []
results_OT_time_breast = []
results_revised_hungarian_time_breast = []

# %% independent case comparison
for sample, l in settings:
    Xvarindex = np.random.randint(0, len(Xvar), sample)
    Yvarindex = np.random.randint(0, len(Yvar), sample)

    Xvarsample = Xvar[Xvarindex][:, 0:5]
    Yvarsample = Yvar[Yvarindex][:, 5:30]

    cost1 = cdist(Xvarsample, np.repeat(Xvarsample, sample, axis=0), 'minkowski', p=l)
    cost2 = cdist(Yvarsample, np.tile(Yvarsample, (sample, 1)), 'minkowski', p=l)
    cost = cost1 + cost2

    costrep = np.repeat(cost, sample, axis=0)


    a = [1 / sample] * sample
    b = [1 / (sample * sample)] * (sample * sample)
    time0 = time.time()
    solution = revised_Hungarian(cost.max() - cost)
    storerevisedHungariantime = time.time() - time0
    results_revised_hungarian_time_breast.append(storerevisedHungariantime)

    time0 = time.time()
    solution = ot.sinkhorn(np.array(a), np.array(b), cost, reg=0.1, acc=0.0001)
    storeOTtime = time.time() - time0
    results_OT_time_breast.append(storeOTtime)

    a = [1 / sample] * sample
    b = [1 / (sample * sample)] * (sample * sample)
    print(sample)

    a = np.reshape([1 / sample] * sample, (sample, 1))
    b = np.reshape([1 / (sample * sample)] * (sample * sample), (sample * sample, 1))

    time0 = time.time()

    # ot.unbalanced.lbfgsb_unbalanced(a,b,cost,reg=0.1,reg_m=0.1)
    uot_sol = sinkhorn_uot(cost, a, b)
    storeUOTtime = time.time() - time0
    results_UOT_time_breast.append(storeUOTtime)
    print(np.shape(cost))

    print(np.sum(solution * cost) - np.sum(uot_sol * cost))

    # solution, storeHungarianoper = track_Hungarian(costrep.max()-costrep)
    # results_hungarian_oper_breast.append(storeHungarianoper)
    #  time0=time.time()
    #  G = nx.DiGraph()
    #  left  = list(np.arange(sample))
    #  right =  list(np.arange(sample*sample)+sample)
    #
    #  G.add_nodes_from(left,demand = -sample)
    #  G.add_nodes_from(right,demand= 1)
    #
    #  source_nodes = list(np.repeat(left,sample*sample))
    #  dest_nodes = right*sample
    #
    #
    #  vfunc = np.vectorize(lambda t: math.floor(t))
    #  costround = vfunc(cost)
    #
    # # Each element of this zip will be
    # # (source[i], dest[i], data[i])
    #  for u,v,d in zip(source_nodes, dest_nodes, costround.ravel()):
    #     G.add_edge(u, v, weight=d)
    #
    #  flowCost, flowDict = nx.network_simplex(G)
    #  storenettime = time.time()-time0
    #  results_net_time_breast.append(storenettime)

    print("running {}/{} experiment".format(i, len(settings)))
    i += 1

# %%intialize parameters before iterations (dependent case
m = np.arange(5, 40, 5)
sample = np.tile(np.repeat(m, 10), 2)
l = np.repeat(np.array([1, 2]), 70)
settings = list(zip(sample, l))

# %%create list to store the results
i = 1
results_UOT_time_z_breast = []
results_OT_time_z_breast = []
results_revised_hungarian_time_z_breast  = []

# %% dependent case
for sample, l in settings:
    Xvarindex = np.random.randint(0, len(Xvar), sample)
    Yvarindex = np.random.randint(0, len(Yvar), sample)

    Xvarsample = Xvar[Xvarindex]
    Yvarsample = Yvar[Yvarindex]
    Zvarsample = Xvarsample[:, 0:5] * Yvarsample[:, 0:5]

    cost1 = cdist(Xvarsample, np.repeat(Xvarsample, sample, axis=0), 'minkowski', p=l)
    cost2 = cdist(Zvarsample, np.tile(Zvarsample, (sample, 1)), 'minkowski', p=l)
    cost = cost1 + cost2

    costrep = np.repeat(cost, sample, axis=0)

    time0 = time.time()
    solution = revised_Hungarian(cost.max() - cost)
    storerevisedHungariantime = time.time() - time0
    results_revised_hungarian_time_z_breast.append(storerevisedHungariantime)

    a = [1 / sample] * sample
    b = [1 / (sample * sample)] * (sample * sample)

    time0 = time.time()
    solution = ot.sinkhorn(np.array(a), np.array(b), cost, reg=0.1, acc=0.0001)
    storeOTtime = time.time() - time0
    results_OT_time_z_breast.append(storeOTtime)

    a = np.reshape([1 / sample] * sample, (sample, 1))
    b = np.reshape([1 / (sample * sample)] * (sample * sample), (sample * sample, 1))
    time0 = time.time()

    # ot.unbalanced.lbfgsb_unbalanced(a,b,cost,reg=0.1,reg_m=0.1)
    uot_sol = sinkhorn_uot(cost, a, b)
    storeUOTtime = time.time() - time0
    results_UOT_time_z_breast.append(storeUOTtime)

    print(np.sum(solution * cost) - np.sum(uot_sol * cost))
    #  time0=time.time()
    #  G = nx.DiGraph()
    #  left  = list(np.arange(sample))
    #  right =  list(np.arange(sample*sample)+sample)
    #
    #  G.add_nodes_from(left,demand = -sample)
    #  G.add_nodes_from(right,demand= 1)
    #
    #  source_nodes = list(np.repeat(left,sample*sample))
    #  dest_nodes = right*sample
    #
    #
    #  vfunc = np.vectorize(lambda t: math.floor(t))
    #  costround = vfunc(cost)
    #
    # # Each element of this zip will be
    # # (source[i], dest[i], data[i])
    #  for u,v,d in zip(source_nodes, dest_nodes, costround.ravel()):
    #     G.add_edge(u, v, weight=d)
    #
    #  flowCost, flowDict = nx.network_simplex(G)
    #  storenettime = time.time()-time0
    #  results_net_time_z_breast.append(storenettime)

    print("running {}/{} experiment".format(i, len(settings)))
    i += 1

####################
# plot of comparison#
####################
# %%
matplotlib.rcParams['lines.linewidth'] = 2.5
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['axes.labelsize'] = 25
matplotlib.rcParams['legend.fontsize'] = 15
matplotlib.rcParams['axes.titlesize'] = 25
matplotlib.rcParams['lines.markersize'] = 6

# %%plotting independent and dependent case wrt time
l = [1, 2]
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))
axes[0].set_ylabel(r'$ln(time)$')
for i in [0, 1]:
    results_revised_hungarian_time_reshape = np.array(results_revised_hungarian_time_z_breast).reshape(2, 70)[
        i].reshape(7, 10)
    results_revised_hungarian_time_worst = results_revised_hungarian_time_reshape.max(axis=1)
    results_revised_hungarian_time_best = results_revised_hungarian_time_reshape.min(axis=1)
    results_revised_hungarian_time_average = results_revised_hungarian_time_reshape.mean(axis=1)

    results_OT_time_reshape = np.array(results_OT_time_z_breast).reshape(2, 70)[
        i].reshape(7, 10)
    results_OT_time_worst = results_OT_time_reshape.max(axis=1)
    results_OT_time_best = results_OT_time_reshape.min(axis=1)
    results_OT_time_average = results_OT_time_reshape.mean(axis=1)

    results_UOT_time_reshape = np.array(results_UOT_time_z_breast).reshape(2, 70)[i].reshape(7, 10)
    results_UOT_time_worst = results_UOT_time_reshape.max(axis=1)
    results_UOT_time_best = results_UOT_time_reshape.min(axis=1)
    results_UOT_time_average = results_UOT_time_reshape.mean(axis=1)
    #
    # axes[i].plot(np.log(m), np.log(results_revised_hungarian_time_average), label='OT modified Hungarian average',
    #              marker='o', color='firebrick')
    # axes[i].plot(np.log(m), np.log(results_revised_hungarian_time_best), label='OT modified Hungarian best', marker='o',
    #              linestyle='-.', color='lightcoral')
    # axes[i].plot(np.log(m), np.log(results_revised_hungarian_time_worst), label='OT modified Hungarian worst', marker='o',
    #              linestyle=':', color='maroon')
    #
    # axes[i].plot(np.log(m), np.log(results_OT_time_average), label='OT Sinkhorn average',
    #              marker='o', color='mediumslateblue')
    # axes[i].plot(np.log(m), np.log(results_OT_time_best), label='OT Sinkhorn best', marker='o',
    #              linestyle='-.', color='plum')
    # axes[i].plot(np.log(m), np.log(results_OT_time_worst), label='OT Sinkhorn worst', marker='o',
    #              linestyle=':', color='indigo')
    #
    # axes[i].plot(np.log(m), np.log(results_UOT_time_average), label='UOT Sinkhorn average', marker='*', color='dodgerblue')
    # axes[i].plot(np.log(m), np.log(results_UOT_time_best), label='UOT Sinkhorn best', marker='*', linestyle='-.',
    #              color='skyblue')
    # axes[i].plot(np.log(m), np.log(results_UOT_time_worst), label='UOT Sinkhorn worst', marker='*', linestyle=':',
    #              color='steelblue')
    axes[i].plot(np.log(m), np.log(results_revised_hungarian_time_average), label='OT modified Hungarian',
                 linestyle='-',marker='o', color='firebrick')
    axes[i].fill_between(np.log(m), np.log(results_revised_hungarian_time_best),
                         np.log(results_revised_hungarian_time_worst), color='lightcoral',alpha=0.4,linestyle='-')

    axes[i].plot(np.log(m), np.log(results_UOT_time_average), label=' UOT Sinkhorn',
                 linestyle='-',marker='o', color='indigo')
    axes[i].fill_between(np.log(m), np.log(results_UOT_time_best),
                         np.log(results_UOT_time_worst), color='darkorchid',alpha=0.4,linestyle='-')

    axes[i].plot(np.log(m), np.log(results_OT_time_average), label='OT Sinkhorn',
                 linestyle='-',marker='o', color='navy')
    axes[i].fill_between(np.log(m), np.log(results_OT_time_best),
                         np.log(results_OT_time_worst), color='lightsteelblue',alpha=0.4,linestyle='-')

    # axes[i].plot(np.log(m), results_revised_hungarian_time_average, label='modified Hungarian average',
    #              marker='o', color='mediumslateblue')
    # axes[i].plot(np.log(m), results_revised_hungarian_time_best, label='modified Hungarian best', marker='o',
    #              linestyle='-.', color='plum')
    # axes[i].plot(np.log(m), results_revised_hungarian_time_worst, label='modified Hungarian worst', marker='o',
    #              linestyle=':', color='indigo')
    #
    # axes[i].plot(np.log(m), results_UOT_time_average, label='UOT average', marker='*', color='dodgerblue')
    # axes[i].plot(np.log(m), results_UOT_time_best, label='UOT best', marker='*', linestyle='-.',
    #              color='skyblue')
    # axes[i].plot(np.log(m), results_UOT_time_worst,label='UOT worst', marker ='*',linestyle=':',color='steelblue')

    axes[i].set_xlabel(r'$ln(sample \ size)$')
    axes[i].set_title(r"dep. case, $p={}$".format(l[i]))
    axes[i].set_ylim(-11, 2)

l = [1, 2]
for i in [0, 1]:
    results_revised_hungarian_time_reshape = np.array(results_revised_hungarian_time_breast).reshape(2, 70)[i].reshape(
        7, 10)
    results_revised_hungarian_time_worst = results_revised_hungarian_time_reshape.max(axis=1)
    results_revised_hungarian_time_best = results_revised_hungarian_time_reshape.min(axis=1)
    results_revised_hungarian_time_average = results_revised_hungarian_time_reshape.mean(axis=1)

    results_OT_time_reshape = np.array(results_OT_time_breast).reshape(2, 70)[i].reshape(
        7, 10)
    results_OT_time_worst = results_OT_time_reshape.max(axis=1)
    results_OT_time_best = results_OT_time_reshape.min(axis=1)
    results_OT_time_average = results_OT_time_reshape.mean(axis=1)

    results_UOT_time_reshape = np.array(results_UOT_time_breast).reshape(2, 70)[i].reshape(7, 10)
    results_UOT_time_worst = results_UOT_time_reshape.max(axis=1)
    results_UOT_time_best = results_UOT_time_reshape.min(axis=1)
    results_UOT_time_average = results_UOT_time_reshape.mean(axis=1)

    # axes[i + 2].plot(np.log(m), np.log(results_revised_hungarian_time_average), label='OT modified Hungarian average',
    #                  marker='o', color='firebrick')
    # axes[i + 2].plot(np.log(m), np.log(results_revised_hungarian_time_best), label='OT modified Hungarian best',
    #                  marker='o', linestyle='-.', color='lightcoral')
    # axes[i + 2].plot(np.log(m), np.log(results_revised_hungarian_time_worst), label='OT modified Hungarian worst',
    #                  marker='o', linestyle=':', color='maroon')
    #
    # #
    # axes[i + 2].plot(np.log(m), np.log(results_OT_time_average), label='OT Sinkhorn average',
    #                  marker='o', color='mediumslateblue')
    # axes[i + 2].plot(np.log(m), np.log(results_OT_time_best), label= 'OT Sinkhorn best',
    #                  marker='o', linestyle='-.', color='plum')
    # axes[i + 2].plot(np.log(m), np.log(results_OT_time_worst), label='OT Sinkhorn worst',
    #                  marker='o', linestyle=':', color='indigo')
    #
    # axes[i + 2].plot(np.log(m), np.log(results_UOT_time_average), label='UOT Sinkhorn average', marker='*', color='dodgerblue')
    # axes[i + 2].plot(np.log(m), np.log(results_UOT_time_best), label='UOT Sinkhorn best', marker='*', linestyle='-.',
    #                  color='skyblue')
    # axes[i + 2].plot(np.log(m), np.log(results_UOT_time_worst), label='UOT Sinkhorn worst', marker='*', linestyle=':',
    #                  color='steelblue')

    axes[i+2].plot(np.log(m), np.log(results_revised_hungarian_time_average), label='OT modified Hungarian',
                 linestyle='-',marker='o', color='firebrick')
    axes[i+2].fill_between(np.log(m), np.log(results_revised_hungarian_time_best),
                         np.log(results_revised_hungarian_time_worst),color='lightcoral',alpha=0.4,linestyle='-')

    axes[i+2].plot(np.log(m), np.log(results_UOT_time_average), label=' UOT Sinkhorn',
                 linestyle='-',marker='o', color='indigo')
    axes[i+2].fill_between(np.log(m), np.log(results_UOT_time_best),
                         np.log(results_UOT_time_worst),color='darkorchid', alpha=0.4, linestyle='-')

    axes[i+2].plot(np.log(m), np.log(results_OT_time_average), label='OT Sinkhorn',
                 linestyle='-',marker='o', color='navy')
    axes[i+2].fill_between(np.log(m), np.log(results_OT_time_best),
                         np.log(results_OT_time_worst),color='lightsteelblue',alpha=0.4, linestyle='-')

    # axes[i+2].plot(np.log(m), results_revised_hungarian_time_average,label='modified Hungarian average', marker ='o',color='mediumslateblue')
    # axes[i+2].plot(np.log(m), results_revised_hungarian_time_best,label='modified Hungarian best', marker ='o',linestyle='-.',color='plum')
    # axes[i+2].plot(np.log(m), results_revised_hungarian_time_worst,label='modified Hungarian worst', marker ='o',linestyle=':',color='indigo')
    #
    # axes[i+2].plot(np.log(m), results_UOT_time_average,label='UOT average', marker ='*',color='dodgerblue')
    # axes[i+2].plot(np.log(m), results_UOT_time_best,label='UOT best', marker ='*',linestyle='-.',color='skyblue')
    # axes[i+2].plot(np.log(m), results_UOT_time_worst,label='UOT worst', marker ='*',linestyle=':',color='steelblue')

    axes[i + 2].set_xlabel(r'$ln(sample \ size)$')
    axes[i + 2].set_title(r"ind. case, $p={}$".format(l[i]))
    axes[i + 2].set_ylim(-11, 2)

fig.tight_layout(rect=[0, 0.08, 1, 1])
handles, labels = axes[0].get_legend_handles_labels()
lgd = fig.legend(
    handles, labels, loc='lower center',
    bbox_to_anchor=(0.5, -0.02), ncol=6)

plt.show()
plt.savefig("Breasttimenet.pdf".format(l[i]), format="pdf")

