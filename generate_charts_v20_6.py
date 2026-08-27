import matplotlib.pyplot as plt
import numpy as np
import os

# Set modern style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

# 데이터 설정 (V20.6 논문용 최종 시스템 검증)
# TP=71, TN=50, FP=5, FN=19
# Confusion Matrix structure:
# [[TN, FP],
#  [FN, TP]]
cm = np.array([[50, 5], [19, 71]])
labels = ['No Swing', 'Swing']

metrics = {
    'Accuracy': 83.45,
    'Precision': 93.42,
    'Recall': 78.89,
    'F1-Score': 85.54
}

# 출력 디렉토리 설정
output_dir = 'plots'  # 차트 저장 폴더 (자동 생성)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 1. Confusion Matrix Heatmap
fig, ax = plt.subplots(figsize=(8, 7))
# Use a more premium color map
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

# Adding color bar with label
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel("Count", rotation=-90, va="bottom", fontsize=12)

ax.set(xticks=np.arange(cm.shape[1]),
       yticks=np.arange(cm.shape[0]),
       xticklabels=labels, yticklabels=labels,
       title='Confusion Matrix',
       ylabel='True Label',
       xlabel='Predicted Label')

# Typography and label adjustments
ax.title.set_fontsize(16)
ax.title.set_fontweight('bold')
ax.xaxis.label.set_fontsize(12)
ax.yaxis.label.set_fontsize(12)

# Text inside cells
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=18, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'confusion_matrix_final.png'), dpi=300, bbox_inches='tight')
plt.close()

# 2. Performance Metrics Bar Chart
names = list(metrics.keys())
values = list(metrics.values())

plt.figure(figsize=(10, 7))
# Sleek, premium color palette
colors = ['#4e79a7', '#f28e2b', '#59a14f', '#edc948']
bars = plt.bar(names, values, color=colors, alpha=0.85, edgecolor='none', width=0.6)

# Display values on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1.5, f'{yval:.2f}%', 
             ha='center', va='bottom', fontsize=13, fontweight='bold', color='#333333')

plt.ylim(0, 115)
plt.xticks(fontsize=15, fontweight='bold')
plt.yticks(fontsize=12)
plt.ylabel('Percentage (%)', fontsize=12, fontweight='bold', labelpad=10)
plt.title('Final System Performance Metrics', fontsize=16, fontweight='bold', pad=25)

# Removing top and right spines for a modern look
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'performance_metrics_final.png'), dpi=300, bbox_inches='tight')
plt.close()

# 3. Result Distribution Pie Chart
error_labels = ['True Positive (Swing)', 'True Negative (No Swing)', 
                'False Positive (Over-call)', 'False Negative (Missed)']
error_counts = [71, 50, 5, 19]
# Premium soft palette
colors = ['#5dabff', '#ffd066', '#ff8585', '#95df95']
explode = (0.02, 0.02, 0.1, 0.1)

plt.figure(figsize=(9, 9))
wedges, texts, autotexts = plt.pie(error_counts, labels=error_labels, autopct='%1.1f%%', 
                                  startangle=140, colors=colors, explode=explode, 
                                  shadow=False, pctdistance=0.8)

plt.setp(autotexts, size=12, weight="bold", color="#222222")
plt.setp(texts, size=11)
plt.title('Classification Result Distribution', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'result_distribution_final.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"최종 시각화 자료가 성공적으로 생성되었습니다: {output_dir}")
