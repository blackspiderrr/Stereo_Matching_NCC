import numpy as np
import cv2

def translation(img, w_shape):
    max = round((w_shape[0]-1) / 2)
    shifted = []
    for i in range(max + 1):
        for j in range(max + 1):
            if i == 0 & j == 0:
                M1 = np.float32([[1, 0, i], [0, 1, j]])
                shifted.append(cv2.warpAffine(img, M1, (img.shape[1], img.shape[0])))
            elif i == 0 & j != 0:
                M1 = np.float32([[1, 0, i], [0, 1, j]])
                M2 = np.float32([[1, 0, i], [0, 1, -j]])
                shifted.append(cv2.warpAffine(img, M1, (img.shape[1], img.shape[0])))
                shifted.append(cv2.warpAffine(img, M2, (img.shape[1], img.shape[0])))
            elif i != 0 & j == 0:
                M1 = np.float32([[1, 0, i], [0, 1, j]])
                M2 = np.float32([[1, 0, -i], [0, 1, j]])
                shifted.append(cv2.warpAffine(img, M1, (img.shape[1], img.shape[0])))
                shifted.append(cv2.warpAffine(img, M2, (img.shape[1], img.shape[0])))
            else:
                M1 = np.float32([[1, 0, i], [0, 1, j]])
                M2 = np.float32([[1, 0, -i], [0, 1, j]])
                M3 = np.float32([[1, 0, -i], [0, 1, -j]])
                M4 = np.float32([[1, 0, i], [0, 1, -j]])
                shifted.append(cv2.warpAffine(img, M1, (img.shape[1], img.shape[0])))
                shifted.append(cv2.warpAffine(img, M2, (img.shape[1], img.shape[0])))
                shifted.append(cv2.warpAffine(img, M3, (img.shape[1], img.shape[0])))
                shifted.append(cv2.warpAffine(img, M4, (img.shape[1], img.shape[0])))
    return np.array(shifted)

def img_sub_avg(img_shifted, avg):
    length, height, width = img_shifted.shape
    tmp_ncc0 = np.zeros([length, height, width])
    for i in range(length):
        tmp_ncc0[i] = img_shifted[i] - avg
    return tmp_ncc0

def NCC(img1_sub_avg, img2_sub_avg, threshold, max_d):
    length, height, width = img1_sub_avg.shape
    ncc_d = np.zeros([height, width])
    ncc_max = np.zeros([height, width])
    for i in range(max_d):
        tmp_ncc1 = np.zeros([length, height, width])
        tmp_ncc2 = np.zeros([height, width])
        tmp_ncc3 = np.zeros([height, width])
        tmp_ncc4 = np.zeros([height, width])
        for j in range(length):
            M1 = np.float32([[1, 0, -i - 1], [0, 1, 0]])
            tmp_ncc1[j] = cv2.warpAffine(img1_sub_avg[j], M1, (img1_sub_avg.shape[2], img1_sub_avg.shape[1]))
        for k in range(length):
            tmp_ncc2 += img2_sub_avg[k] * tmp_ncc1[k]
            tmp_ncc3 += pow(img2_sub_avg[k], 2)
            tmp_ncc4 += pow(tmp_ncc1[k], 2)
        tmp_ncc = tmp_ncc2 / np.sqrt(tmp_ncc3 * tmp_ncc4)
        for m in range(height):
            for n in range(width):
                if (tmp_ncc[m, n] > threshold) & (tmp_ncc[m, n] > ncc_max[m, n]):
                    ncc_max[m, n] = tmp_ncc[m, n]
                    ncc_d[m, n] = i
    return ncc_d

if __name__ == "__main__":
    size = 5
    img1 = cv2.imread('D:\py\Semantic Segmentation\Stereo Matching\image1.png', cv2.CV_8UC1)
    img2 = cv2.imread('D:\py\Semantic Segmentation\Stereo Matching\image2.png', cv2.CV_8UC1)
    img1_avg = cv2.blur(img1, (size, size))
    img2_avg = cv2.blur(img2, (size, size))
    img1_float = img1.astype(np.float32)
    img2_float = img2.astype(np.float32)
    img1_avg = img1_avg.astype(np.float32)
    img2_avg = img2_avg.astype(np.float32)
    img1_shifted = translation(img1_float, [size, size])
    img2_shifted = translation(img2_float, [size, size])
    img1_sub_avg = img_sub_avg(img1_shifted, img1_avg)
    img2_sub_avg = img_sub_avg(img2_shifted, img2_avg)
    ncc_d = NCC(img1_sub_avg, img2_sub_avg, threshold=0.5, max_d=64)
    disparity = cv2.normalize(ncc_d, ncc_d, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imshow('Left_Img', img1)
    cv2.imshow('Right_Img', img2)
    cv2.imshow('Disparity', disparity)
    cv2.waitKey(0)  # 等待按键按下
    cv2.destroyAllWindows()  # 清除所有窗口
