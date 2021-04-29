from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import copy



parser = argparse.ArgumentParser(description='Calcualte the detection AUC of Neural Cleanse')
parser.add_argument('--dataset', type=str, default='zzz',
                    help='dataset to use')
parser.add_argument('--network-arch', type=str, help='network architecture')
parser.add_argument('--try-version', type=str, help='experiment setting')

args = parser.parse_args()

anomaly_index_save_path = './neural_cleanse_anomaly_index/' + '%s_%s_%s' % (args.network_arch, args.dataset.lower(), 'clean')
result = pickle.load(open(anomaly_index_save_path, "rb"))

value_list = []
label_list = []
for anomaly_index_list in result:
    # print(anomaly_index_list)
    value_list.append(max(anomaly_index_list))
    label_list.append(0)

anomaly_index_save_path = './neural_cleanse_anomaly_index/' + '%s_%s_%s' % (args.network_arch, args.dataset.lower(), args.try_version)
result = pickle.load(open(anomaly_index_save_path, "rb"))

for anomaly_index_list in result:
    # print(anomaly_index_list)
    value_list.append(max(anomaly_index_list))
    label_list.append(1)

value_list = np.array(value_list)
label_list = np.array(label_list)

fpr, tpr, thresh = roc_curve(label_list, value_list)

# for i, j, k in zip(fpr, tpr, thresh):
#     print(i, j, k)

previous_values = []
for i, j, k in zip(fpr, tpr, thresh):
    if i > 0.1:
        result = copy.deepcopy(previous_values)
        break
    previous_values = [i, j, k]

auc = roc_auc_score(label_list, value_list)

# print("labels:", label_list)
# print("anomaly indexes:", value_list)
# print("threshold: %.3f, fpr: %.3f, tpr: %.3f" % (result[2], result[0], result[1]))
print('################################')
print(anomaly_index_save_path)
# print('AUC is %.3f' % round(auc, 3))
print('AUC: %.3f; TPR:%.3f' % (round(auc, 3), round(result[1],3)))
print('################################')