import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.switch_backend('TkAgg')

# ====================== 引导滤波 ======================
def guided_filter(I, p, r, eps):
    ksize = (2 * r + 1, 2 * r + 1)
    mean_I = cv2.boxFilter(I, cv2.CV_64F, ksize)
    mean_p = cv2.boxFilter(p, cv2.CV_64F, ksize)
    mean_Ip = cv2.boxFilter(I * p, cv2.CV_64F, ksize)
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(I * I, cv2.CV_64F, ksize)
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_64F, ksize)
    mean_b = cv2.boxFilter(b, cv2.CV_64F, ksize)

    q = mean_a * I + mean_b
    return q

# ====================== 细节保留的多尺度 Retinex ======================
def detail_msr(img, scales=[15, 40, 80], weights=[0.3, 0.4, 0.3]):
    img = img.astype(np.float64) / 255.0
    msr_sum = np.zeros_like(img)

    for c in range(3):
        channel = img[:, :, c]
        msr_ch = np.zeros_like(channel)
        for s, w in zip(scales, weights):
            L = guided_filter(channel, channel, r=s, eps=0.01)
            R = channel / (L + 1e-6)
            msr_ch += w * R
        msr_sum[:, :, c] = msr_ch

    for c in range(3):
        vmin, vmax = np.percentile(msr_sum[:, :, c], 1), np.percentile(msr_sum[:, :, c], 99)
        msr_sum[:, :, c] = np.clip((msr_sum[:, :, c] - vmin) / (vmax - vmin + 1e-6), 0, 1)
    return msr_sum

# ====================== 亮度补偿 ======================
def brightness_compensate(original, enhanced):
    orig_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY).mean() / 255.0
    enh_gray = cv2.cvtColor((enhanced * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).mean() / 255.0
    if enh_gray < orig_gray:
        gamma = np.log(orig_gray) / np.log(enh_gray + 1e-6)
        gamma = np.clip(gamma, 0.5, 2.0)
        enhanced = np.power(enhanced, gamma)
    return enhanced

# ====================== 多尺度细节增强（核心改进）======================
def multi_scale_detail_enhance(img, amount=0.45):
    """
    提取三个尺度的高频细节并叠加，显著提升清晰度
    amount 建议 0.3~0.6
    """
    img_f = img.astype(np.float64)
    base = cv2.GaussianBlur(img_f, (0, 0), 3.0)
    detail1 = img_f - base
    detail2 = base - cv2.GaussianBlur(base, (0, 0), 5.0)
    detail3 = cv2.GaussianBlur(base, (0, 0), 5.0) - cv2.GaussianBlur(base, (0, 0), 10.0)
    detail = detail1 * 0.6 + detail2 * 0.3 + detail3 * 0.1
    enhanced = img_f + amount * detail
    return np.clip(enhanced, 0, 255).astype(np.uint8)

# ====================== 主增强流程 ======================
def retinex_low_light_restore(img, denoise_strength=4, detail_amount=0.45):
    # 1. 极轻去噪（保留纹理）
    denoised = cv2.fastNlMeansDenoisingColored(img, None, denoise_strength,
                                               denoise_strength, 7, 15)
    # 2. 引导滤波 Retinex（边缘清晰）
    enhanced = detail_msr(denoised)

    # 3. 亮度补偿（绝不变暗）
    enhanced = brightness_compensate(img, enhanced)

    # 4. 转 uint8
    final = (np.clip(enhanced, 0, 1) * 255).astype(np.uint8)

    # 5. 多尺度细节增强（代替单次锐化，清晰度显著提升）
    final = multi_scale_detail_enhance(final, amount=detail_amount)

    return final

# ====================== 主程序 ======================
if __name__ == "__main__":
    img_path = "C:/Users/fy/Desktop/QQ20260529-184209.png"
    img = cv2.imread(img_path)
    if img is None:
        print("错误：无法读取图像")
        exit()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 参数：去噪4（可降至2），细节增强0.45（可增至0.6）
    enhanced_img = retinex_low_light_restore(img, denoise_strength=4, detail_amount=0.45)

    print(f"原始平均亮度：{np.mean(img):.1f}，增强后：{np.mean(enhanced_img):.1f}")

    # 显示与保存
    plt.figure(figsize=(15,5))
    plt.subplot(131), plt.imshow(img), plt.title("原始低光图像"), plt.axis("off")
    msr_only = detail_msr(img)
    plt.subplot(132), plt.imshow(msr_only), plt.title("引导滤波 MSR"), plt.axis("off")
    plt.subplot(133), plt.imshow(enhanced_img), plt.title("最终超清结果"), plt.axis("off")
    plt.tight_layout()
    plt.savefig('low_light_enhancement_ultraclear.png', dpi=300, bbox_inches='tight')
    cv2.imwrite('enhanced_ultraclear.png', cv2.cvtColor(enhanced_img, cv2.COLOR_RGB2BGR))
    plt.show()
