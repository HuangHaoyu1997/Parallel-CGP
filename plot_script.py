import numpy as np
import matplotlib.pyplot as plt


dir = '/home/hhy/Parallel-CGP/results/7月7日-pr_ratio搜索-40generation/'
with open(dir+'log-Oil-2022-07-06-03-30-24.txt', 'r') as f:
    raw_data = f.readlines()

scores = []
for d in raw_data:
    score = float(d.split('-')[1])
    scores.append(score)


plt.rcParams["font.size"] = 12
plt.rcParams['font.family'] = 'Times New Roman'
plt.grid(visible=True, which='major', linestyle='-')
plt.grid(visible=True, which='minor', linestyle='--', alpha=0.5)
plt.minorticks_on()

plt.plot(scores, 
        'o-.', 
        markersize=4,
        # color='limegreen', 
        # label='ratio loss', 
        linewidth=1,
        )

plt.xlabel('Generation'); plt.ylabel('Loss')
# plt.savefig('ratio_mech_GA_search'+'.pdf', dpi=300, format='pdf',bbox_inches='tight')
plt.savefig(dir+'pr_ratio_mech_CGP_search'+'.svg', dpi=300, format='svg',bbox_inches='tight')