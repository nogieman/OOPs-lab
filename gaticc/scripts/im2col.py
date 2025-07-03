import numpy as np
import sys
import os
from PIL import Image

# image: path to image
# Imagenet preprocessing function
def preprocess(image):
    if not os.path.exists(image):
        raise OSError("File not found: {}".format(image))
    img = Image.open(image)
    # resize to (256,256)
    img = img.resize((256,256))
    img = np.array(img.convert('RGB'))
    # scale b/w 0 and 1
    img = img / 255.
    # take a (224,224) center crop of the image
    h, w = img.shape[0], img.shape[1]
    y0 = (h - 224) // 2
    x0 = (w - 224) // 2
    img = img[y0 : y0+224, x0 : x0+224, :]
    # Normalize (values obtained from millions of imagenet images)
    img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    img = np.transpose(img, axes=[2, 0, 1])
    img = img.astype(np.float32)
    return img

def preproc_npy(file_path):
    if not os.file.exists(file_path):
        raise FileNotFoundError(f"Could not find {file_path}")

    arr = np.load(file_path)
    return arr

def im2col(img, dims, kdims, stride, pad):
    """ generate an image where after im2col the columns have sequential numbers """
    def im2col_single_channel(img, dims, kdims, stride, pad):
        ret = np.empty([dims[0] * dims[1], kdims[0] * kdims[1]])
        assert len(dims) == 2
        assert len(kdims) == 2
        assert len(stride) == 2
        out_height = (dims[0] - kdims[0] + 2 * pad[0]) // stride[0] + 1
        out_width = (dims[1] - kdims[1] + 2 * pad[0]) // stride[1] + 1
        
        img = np.pad(img, pad)
        # Initialize output array
        ret = np.zeros((out_height * out_width, kdims[0] * kdims[1]))
        out_id = 0
        for i in range(0, dims[0]-2, stride[0]):
            for j in range(0, dims[1]-2, stride[1]):
                arr = []
                for k in range(kdims[0]):
                    for l in range(kdims[1]):
                        row = i + k
                        col = j + l
                        arr.append(img[row,col])
                ret[out_id] = arr
                out_id += 1
        return ret
    ret_list = []
    for i in range(img.shape[0]):
        ret_list.append(im2col_single_channel(img[i], dims, kdims, stride, pad))
    return np.array(ret_list)


if __name__ == "__main__":
    col = im2col(preprocess(sys.argv[1]), [224,224], [3,3], [1,1], [1,1])
    print(col[0].flatten()[:60])
