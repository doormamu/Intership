import numpy as np 
def get_bayer_masks(n_rows, n_cols):
    """
    :param n_rows: `int`, number of rows
    :param n_cols: `int`, number of columns

    :return:
        `np.array` of shape `(n_rows, n_cols, 3)` and dtype `np.bool_`
        containing red, green and blue Bayer masks
    """
    r_p = np.array([[0,1],[0,0]], dtype=bool)
    g_p = np.array([[1,0],[0,1]], dtype=bool)
    b_p = np.array([[0,0],[1,0]], dtype=bool)
    r_m = np.tile(r_p, (n_rows//2 + 1, n_cols//2 + 1))[:n_rows,:n_cols]
    g_m = np.tile(g_p, (n_rows//2 + 1, n_cols//2 + 1))[:n_rows,:n_cols]
    b_m = np.tile(b_p, (n_rows//2 + 1, n_cols//2 + 1))[:n_rows,:n_cols]
    return np.dstack((r_m, g_m, b_m))


def get_colored_img(raw_img):
    """
    :param raw_img:
        `np.array` of shape `(n_rows, n_cols)` and dtype `np.uint8`,
        raw image

    :return:
        `np.array` of shape `(n_rows, n_cols, 3)` and dtype `np.uint8`,
        each channel contains known color values or zeros
        depending on Bayer masks
    """
    nr, nc = raw_img.shape
    masks = get_bayer_masks(nr, nc)
    colored = np.zeros((nr, nc, 3), dtype=raw_img.dtype)
    for i in range(3):
        colored[..., i] = raw_img * masks[..., i]
    return colored


def get_raw_img(colored_img):
    """
    :param colored_img:
        `np.array` of shape `(n_rows, n_cols, 3)` and dtype `np.uint8`,
        colored image

    :return:
        `np.array` of shape `(n_rows, n_cols)` and dtype `np.uint8`,
        raw image as captured by camera
    """
    nr, nc, _ = colored_img.shape
    masks = get_bayer_masks(nr, nc)
    raw = np.zeros((nr, nc), dtype=colored_img.dtype)
    for i in range(3):
        raw += colored_img[..., i] * masks[..., i]
    return raw

def bilinear_interpolation(raw_img):
    """
    :param raw_img:
        `np.array` of shape `(n_rows, n_cols)` and dtype `np.uint8`,
        raw image

    :return:
        `np.array` of shape `(n_rows, n_cols, 3)`, and dtype `np.uint8`,
        result of bilinear interpolation
    """
    nr, nc = raw_img.shape
    col = get_colored_img(raw_img)
    masks = get_bayer_masks(nr, nc)
    res = np.zeros_like(col, dtype=np.float32)
    pad = ((1,1),(1,1))

    for i in range(3):
        ch = col[...,i].astype(np.float32)
        m = masks[...,i].astype(np.float32)
        ch_p = np.pad(ch, pad, 'reflect')
        m_p = np.pad(m, pad, 'reflect')

        s = (ch_p[:-2,:-2] + ch_p[:-2,1:-1] + ch_p[:-2,2:] +
             ch_p[1:-1,:-2] + ch_p[1:-1,1:-1] + ch_p[1:-1,2:] +
             ch_p[2:,:-2] + ch_p[2:,1:-1] + ch_p[2:,2:])
        c = (m_p[:-2,:-2] + m_p[:-2,1:-1] + m_p[:-2,2:] +
             m_p[1:-1,:-2] + m_p[1:-1,1:-1] + m_p[1:-1,2:] +
             m_p[2:,:-2] + m_p[2:,1:-1] + m_p[2:,2:])
        c[c==0] = 1
        ch_new = s / c

        ch_new[m == 1] = ch[m == 1]

        res[...,i] = ch_new

    return np.clip(res, 0, 255).astype(raw_img.dtype)


def improved_interpolation(raw_img):
    """
    :param raw_img:
        `np.array` of shape `(n_rows, n_cols)` and dtype `np.uint8`, raw image

    :return:
        `np.array` of shape `(n_rows, n_cols, 3)` and dtype `np.uint8`,
        result of improved interpolation
    """
    h, w = raw_img.shape[:2]
    masks = get_bayer_masks(h, w)
    base_img = get_colored_img(raw_img).astype(np.float32)
    raw_f = raw_img.astype(np.float32)

    # нормировка ядра по сумме ненулевых элементов
    def kernel_coeffs(p):
        k = np.array(p, dtype=np.float32)
        return k / k[k != 0].sum()

    # простая 5x5 свертка (чистый numpy)
    def convolve5x5(img, k):
        pad = np.pad(img, ((2,2),(2,2)), mode='reflect')
        out = np.zeros_like(img)
        for i in range(5):
            for j in range(5):
                if k[i,j] != 0:
                    out += pad[i:i+h, j:j+w] * k[i,j]
        return out

    # маска по четности индексов
    def pos_mask(rp, cp):
        r = np.arange(h)[:, None]
        c = np.arange(w)[None, :]
        return ((r & 1) == rp) & ((c & 1) == cp)

    # обновление канала по маске
    def update(ch, mask, k):
        f = convolve5x5(raw_f, k)
        ch[mask] = f[mask]
        return ch

    # ядра из статьи Malvar
    k_g_cross = kernel_coeffs([
        [0,0,-1,0,0],
        [0,0,2,0,0],
        [-1,2,4,2,-1],
        [0,0,2,0,0],
        [0,0,-1,0,0],
    ])
    k_rb_vert = kernel_coeffs([
        [0,0,0.5,0,0],
        [0,-1,0,-1,0],
        [-1,4,5,4,-1],
        [0,-1,0,-1,0],
        [0,0,0.5,0,0],
    ])
    k_rb_horiz = kernel_coeffs([
        [0,0,-1,0,0],
        [0,-1,4,-1,0],
        [0.5,0,5,0,0.5],
        [0,-1,4,-1,0],
        [0,0,-1,0,0],
    ])
    k_rb_diag = kernel_coeffs([
        [0,0,-1.5,0,0],
        [0,2,0,2,0],
        [-1.5,0,6,0,-1.5],
        [0,2,0,2,0],
        [0,0,-1.5,0,0],
    ])

    out = base_img.copy()
    # позиции по шаблону GR/BG
    r_loc = pos_mask(0,1)
    b_loc = pos_mask(1,0)
    g_even = pos_mask(0,0)
    g_odd = pos_mask(1,1)

    # G канал
    out[...,1] = update(out[...,1], r_loc, k_g_cross)      # исправление: добавлена корректная свертка G на R
    out[...,1] = update(out[...,1], b_loc, k_g_cross)      # исправление: добавлена корректная свертка G на B

    # R канал
    out[...,0] = update(out[...,0], g_even, k_rb_vert)     # исправление: вертикальные соседи G
    out[...,0] = update(out[...,0], g_odd, k_rb_horiz)     # исправление: горизонтальные соседи G
    out[...,0] = update(out[...,0], b_loc, k_rb_diag)      # исправление: диагональные соседи B

    # B канал
    out[...,2] = update(out[...,2], g_even, k_rb_horiz)    # исправление: горизонтальные соседи G
    out[...,2] = update(out[...,2], g_odd, k_rb_vert)      # исправление: вертикальные соседи G
    out[...,2] = update(out[...,2], r_loc, k_rb_diag)      # исправление: диагональные соседи R

    return np.clip(out, 0, 255).astype(np.uint8)




















def compute_psnr(img_pred, img_gt):
    """
    :param img_pred:
        `np.array` of shape `(n_rows, n_cols, 3)` and dtype `np.uint8`,
        predicted image
    :param img_gt:
        `np.array` of shape `(n_rows, n_cols, 3)` and dtype `np.uint8`,
        ground truth image

    :return:
        `float`, PSNR metric
    """
    # исправление: переводим в float для точных вычислений
    pred = img_pred.astype(np.float64)
    gt = img_gt.astype(np.float64)

    diff = pred - gt
    mse = np.mean(diff ** 2)
    if mse == 0:
        # исправление: по заданию нужно вызывать исключение при MSE == 0
        raise ValueError("MSE is zero, PSNR undefined")

    max_val = gt.max()
    psnr = 10 * np.log10((max_val ** 2) / mse)
    return psnr



if __name__ == "__main__":
    from PIL import Image

    raw_img_path = "tests/04_unittest_bilinear_img_input/02.png"
    raw_img = np.array(Image.open(raw_img_path))

    img_bilinear = bilinear_interpolation(raw_img)
    Image.fromarray(img_bilinear).save("bilinear.png")

    img_improved = improved_interpolation(raw_img)
    Image.fromarray(img_improved).save("improved.png")
