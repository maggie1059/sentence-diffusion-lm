from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

model = SentenceTransformer('all-MiniLM-L6-v2')

# filenames = ['./gt.txt', './gpt.txt', './dlm.npy']
# files = [open('./gt.txt', 'r'), open('./gpt.txt', 'r'), np.load('./dlm.npy')]
# for rows in zip(*files):
#     gt = rows[0].split('__eou__')[:-1]
#     num_sents = min(len(gt), 4)
#     gpt = rows[1].split('__eou__')[:num_sents]

#     dlm = rows[2].split('__eou__')[:num_sents]

    
gt_embeds = []
# indices = []
with open('./gt.txt', 'r') as f:
    for idx, row in enumerate(f):
        if len(gt_embeds) == 100: break
        gt = row.split('__eou__')[:-1]
        # if len(gt) < 4: continue
        # indices.append(idx)
        if len(gt) < 4:
            gt += ['[PAD]' for _ in range(4 - len(gt))]
        gt = gt[:4]
        gt_e = model.encode(gt)
        gt_e = np.asarray(gt_e)
        # print(np.shape(gt_e))
        gt_embeds.append(gt_e)
gt_embeds = np.asarray(gt_embeds)

gpt_embeds = []

with open('./gpt.txt', 'r') as f:
    for idx, row in enumerate(f):
        if idx == 100: break
        # if idx in indices:
        gt = row.split('__eou__')[:-1]
        if len(gt) < 4:
            gt += ['[PAD]' for _ in range(4 - len(gt))]
        gt = gt[:4]
        gt_e = model.encode(gt)
        gpt_embeds.append(gt_e)
gpt_embeds = np.asarray(gpt_embeds)

dlm = np.load('./dlm_outputs4.npy')
dlm_embeds = []

for i in range(100):
    row = dlm[i]
    row = row[0, :4]
    dlm_embeds.append(row)
dlm_embeds = np.asarray(dlm_embeds)

print("gt: ", np.shape(gt_embeds))
print("gtp: ", np.shape(gpt_embeds))
print("dlm: ", np.shape(dlm_embeds))

def create_plot(gt, gpt, dlm):
    pca = PCA(n_components=2)
    gt = gt.reshape(-1, 384)
    gpt = gpt.reshape(-1, 384)
    dlm = dlm.reshape(-1, 384)
    all_data = np.concatenate((gt, gpt, dlm))
    all_pca = pca.fit_transform(all_data)
    x = all_pca[:,0]
    y = all_pca[:,1]
    print("x shape: ", np.shape(x))

    # gt_pca = pca.fit_transform(gt)
    # gpt_pca = pca.fit_transform(gpt)
    # dlm_pca = pca.fit_transform(dlm)
    colors = ['r'] * 100 + ['b'] * 100 + ['g'] * 100
    # gt_x = gt_pca[:,0]
    # gpt_x = gpt_pca[:,0]
    # dlm_x = dlm_pca[:,0]
    # gt_y = gt_pca[:,1]
    # gpt_y = gpt_pca[:,1]
    # dlm_y = dlm_pca[:,1]
    # x = np.concatenate((gt_x, gpt_x, dlm_x))
    # y = np.concatenate((gt_y, gpt_y, dlm_y))
    # plt.scatter(x, y, c=colors, alpha=0.2, label=colors)
    gt = plt.scatter(x[:100], y[:100], c=colors[:100], alpha=0.2)
    gpt = plt.scatter(x[100:200], y[100:200], c=colors[100:200], alpha=0.2)
    dlm = plt.scatter(x[200:], y[200:], c=colors[200:], alpha=0.2)
    plt.legend((gt, gpt, dlm),
           ('Ground Truth', 'GPT2', 'Diffusion'),
           scatterpoints=1,
           loc='lower left')
    plt.title('All Embeddings (Sentence 1)')
    plt.savefig('embeds_0.png', bbox_inches='tight')
    plt.show()

def compare_cos(gt1, gpt1, dlm1):
    index = 3
    avg_gpt = 0 #np.zeros((4,))
    avg_dlm = 0 #np.zeros((4,))
    for i in range(100):
        gt = gt1[i, index, :].reshape(1, -1)
        gpt = gpt1[i, index, :].reshape(1, -1)
        dlm = dlm1[i, index, :].reshape(1, -1)
        gt_gpt = cosine_similarity(gt, gpt)
        # print("gt gpt: ", gt_gpt)
        gt_dlm = cosine_similarity(gt, dlm)
        avg_gpt += gt_gpt
        avg_dlm += gt_dlm
    avg_gpt /= 100
    avg_dlm /= 100
    print("avg gpt 3: ", avg_gpt)
    print("avg dlm 3: ", avg_dlm)

def compare_cos2(gt1):
    avgs = np.zeros((3,))
    for i in range(100):
        for index in range(3):
            gt = gt1[i, index, :].reshape(1, -1)
            gt2 = gt1[i, index+1, :].reshape(1, -1)
            gt_gpt = cosine_similarity(gt, gt2)
            avgs[index] += gt_gpt
        avgs /= 100
    print("avgs: ", avgs)

def compute_mse(gt, gpt, dlm):
    avg_gpt = 0
    avg_dlm = 0
    for i in range(100):
        for j in range(1, 4):
            gt_gpt = mean_squared_error(gt[i, j, :], gpt[i, j, :])
            gt_dlm = mean_squared_error(gt[i, j, :], dlm[i, j, :])
            avg_gpt += gt_gpt
            avg_dlm += gt_dlm
    avg_gpt /= 300
    avg_dlm /= 300
    print("mse gpt: ", avg_gpt)
    print("mse dlm: ", avg_dlm)

# create_plot(gt_embeds[:,0,:], gpt_embeds[:,0,:], dlm_embeds[:,0,:])
# compare_cos(gt_embeds, gpt_embeds, dlm_embeds)
# compare_cos2(dlm_embeds)
compute_mse(gt_embeds, gpt_embeds, dlm_embeds)