"""Display initialization candidates and expert similarity on diagnostic coordinates."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

RUN = Path(__file__).resolve().parent
report = json.loads((RUN / 'evidence_verification.json').read_text())
keys = report['expert_order']
names = [key.replace('_checkpoint_300', '').upper() for key in keys]
left = np.array([[report['experts'][key]['candidate_cosine_4090'],
                  report['experts'][key]['candidate_cosine_3090']] for key in keys])
right = np.array(report['pairwise_cosine'])
font = Path('/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc')
if font.exists():
    font_manager.fontManager.addfont(str(font))
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
plt.rcParams.update({'font.size': 10, 'axes.unicode_minus': False, 'pdf.fonttype': 42})
fig, axes = plt.subplots(1, 2, figsize=(14, 8), gridspec_kw={'width_ratios': [1, 3.2]})
for ax, matrix in zip(axes, [left, right]):
    image = ax.imshow(matrix, vmin=-1, vmax=1, cmap='RdBu_r', aspect='auto')
    ax.set_yticks(range(len(names)), names)
    for (row, col), value in np.ndenumerate(matrix):
        ax.text(col, row, f'{value:.2f}', ha='center', va='center', fontsize=8,
                color='white' if abs(value) > .65 else 'black')
    ax.axhline(8.5, color='#252525', linewidth=1.8)
axes[0].set_xticks([0, 1], ['候选A\n4090重建', '候选B\n3090重建'])
axes[0].set_title('专家与初始化候选的余弦相似度', pad=14)
axes[1].set_xticks(range(len(names)), names, rotation=55, ha='right')
axes[1].axvline(8.5, color='#252525', linewidth=1.8)
axes[1].set_title('专家之间的余弦相似度', pad=14)
fig.subplots_adjust(left=.09, right=.89, top=.82, bottom=.22, wspace=.38)
fig.colorbar(image, cax=fig.add_axes([.91, .25, .015, .53]), label='Cosine similarity')
fig.suptitle('URS初始化核查：支持9＋2分组的参数证据', fontsize=17, y=.97)
fig.text(.09, .9, '仅比较两种候选不同的348,160个坐标；不使用测试表现挑选base。', fontsize=11)
fig.text(.09, .08, '精确事实：未更新随机参数匹配。分组依据：已更新参数的相似度，属于强统计推断。', fontsize=10)
fig.text(.09, .045, 'GPU名称只标记重建环境；不能据此认定历史训练硬件或已获得完整历史epoch0。', fontsize=10)
for suffix in ['png', 'pdf']:
    output = RUN / f'initialization_evidence.{suffix}'
    fig.savefig(output, dpi=180, bbox_inches='tight')
    print(output)
plt.close(fig)
