import matplotlib.pyplot as plt
import numpy as np
import os

# 데이터 설정
# TP=74, TN=49, FP=6, FN=16
cm = np.array([[49, 6], [16, 74]])
labels = ['No Swing', 'Swing']

metrics = {
    'Accuracy': 84.83,
    'Precision': 92.50,
    'Recall': 82.22,
    'F1-Score': 87.06
}

# 출력 디렉토리 설정
output_dir = 'plots'  # 차트 저장 폴더 (자동 생성)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 1. Confusion Matrix Heatmap (Matplotlib only)
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)
ax.set(xticks=np.arange(cm.shape[1]),
       yticks=np.arange(cm.shape[0]),
       xticklabels=labels, yticklabels=labels,
       title='Confusion Matrix',
       ylabel='True Label',
       xlabel='Predicted Label')

# 텍스트 추가
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=16, fontweight='bold')

fig.tight_layout()
plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300)
plt.close()

# 2. Performance Metrics Bar Chart
names = list(metrics.keys())
values = list(metrics.values())

plt.figure(figsize=(10, 6))
# Matplotlib colors
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
bars = plt.bar(names, values, color=colors, alpha=0.8, edgecolor='black')

# 막대 위에 값 표시
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.ylim(0, 110)
plt.ylabel('Score (%)', fontsize=12, fontweight='bold')
plt.title('System Performance Metrics (%)', fontsize=14, fontweight='bold', pad=20)
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'performance_metrics.png'), dpi=300)
plt.close()

# 3. Error Analysis Pie Chart
error_labels = ['True Positive', 'True Negative', 'False Positive', 'False Negative']
error_counts = [74, 49, 6, 16]
colors = ['#aec7e8', '#ffbb78', '#98df8a', '#ff9896']
explode = (0.05, 0.05, 0.1, 0.1)

plt.figure(figsize=(8, 8))
plt.pie(error_counts, labels=error_labels, autopct='%1.1f%%', startangle=140, 
        colors=colors, explode=explode, shadow=False, textprops={'fontsize': 12})
plt.title('Classification Result Distribution', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'result_distribution.png'), dpi=300)
plt.close()

print(f"이미지들이 성공적으로 저장되었습니다: {output_dir}")
