import argparse
import os

import matplotlib
import numpy as np
import scipy.io
import torch
from torchvision import datasets

matplotlib.use("agg")
import matplotlib.pyplot as plt


#####################################################################
# Show result
def imshow(path, title=None):
    """Imshow for Tensor."""
    im = plt.imread(path)
    plt.imshow(im)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def sort_img(qf, ql, qc, gf, gl, gc):
    query = qf.view(-1, 1)
    # print(query.shape)
    score = torch.mm(gf, query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    # predict index
    index = np.argsort(score)  # from small to large
    index = index[::-1]
    # index = index[0:2000]
    # good index
    query_index = np.argwhere(gl == ql)
    # same camera
    camera_index = np.argwhere(gc == qc)

    # good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gl == -1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1)

    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]
    return index


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--nums", default=100, type=int, help="need nums query to rank")
    parser.add_argument(
        "--output", default="./vis/ranklist/baseline", type=str, help="save ranklist figure"
    )
    parser.add_argument(
        "--input", default="./vis/ranklist/baseline", type=str, help="save ranklist figure"
    )
    opts = parser.parse_args()

    os.makedirs(opts.output, exist_ok=True)

    result = scipy.io.loadmat(opts.input)
    query_feature = torch.FloatTensor(result["query_f"])
    query_cam = result["query_cam"][0]
    query_label = result["query_label"][0]
    query_img_paths = result["query_img_paths"]
    gallery_feature = torch.FloatTensor(result["gallery_f"])
    gallery_cam = result["gallery_cam"][0]
    gallery_label = result["gallery_label"][0]
    gallery_img_paths = result["gallery_img_paths"]

    query_feature = query_feature.cuda()
    gallery_feature = gallery_feature.cuda()

    for i in range(opts.nums):
        index = sort_img(
            query_feature[i],
            query_label[i],
            query_cam[i],
            gallery_feature,
            gallery_label,
            gallery_cam,
        )

        ########################################################################
        # Visualize the rank result

        query_path = query_img_paths[i].strip()
        cur_query_label = query_label[i]
        print(query_path)
        print("Top 10 images are as follow:")
        try:  # Visualize Ranking Result
            # Graphical User Interface is needed
            fig = plt.figure(figsize=(16, 4))
            ax = plt.subplot(1, 11, 1)
            ax.axis("off")
            imshow(query_path, "query")
            for j in range(10):
                ax = plt.subplot(1, 11, j + 2)
                ax.axis("off")
                img_path = gallery_img_paths[index[j]].strip()
                label = gallery_label[index[j]]
                imshow(img_path)
                if label == cur_query_label:
                    ax.set_title("%d" % (j + 1), color="green")
                else:
                    ax.set_title("%d" % (j + 1), color="red")
                print(img_path)
        except RuntimeError:
            print("Skip!!!")

        fig.savefig(f"./{opts.output}/{i}.png")
        fig.clear()
