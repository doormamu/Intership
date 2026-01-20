import numpy as np
#import cv2 as cv

# Read the implementation of the align_image function in pipeline.py
# to see, how these functions will be used for image alignment.


def extract_channel_plates(raw_img, crop):
    #unaligned_rgb = raw_img.reshape(-1, 3)
    #unaligned_rgb = (np.zeros(0), np.zeros(0), np.zeros(0))

    height = raw_img.shape[0]
    '''
    if height % 3 != 0:
        raw1 = np.delete(raw_img, -(height % 3), 0) #?
    '''
    if height % 3 != 0:
        b_img = raw_img[ : height//3]
        g_img = raw_img[height//3 : 2*(height//3)]
        r_img = raw_img[2*(height//3) : -(height % 3)]
    else:
        b_img = raw_img[ : height//3]
        g_img = raw_img[height//3 : 2*(height//3)]
        r_img = raw_img[2*(height//3) :]

    assert (b_img.size == g_img.size) and (g_img.size == r_img.size)
    newh = r_img.shape[0]
    neww = r_img.shape[1]

    if crop == False:
        unaligned_rgb = (r_img, g_img, b_img)
        coords = (np.array([2*(height//3),0]), np.array([height//3 ,0]), np.array([0,0]))
    else:
        r_img1 = r_img[newh//10: -(newh//10), neww//10: -(neww//10)]
        g_img1 = g_img[newh//10: -(newh//10), neww//10: -(neww//10)]
        b_img1 = b_img[newh//10: -(newh//10), neww//10: -(neww//10)]
        unaligned_rgb = (r_img1, g_img1, b_img1)
        coords = (np.array([(newh*2 + newh//10),neww//10]), np.array([(newh//10 + newh),neww//10]), np.array([newh//10,neww//10]))

    return unaligned_rgb, coords


def find_relative_shift_pyramid(img_a, img_b):

    def find_metric(region_a, region_b, metric):
        
        ### Args:
        #область пересечения изображений, метрика 1 (MSE) или 2 (корелляция)
        
        if metric == 1:
            return np.mean((region_a - region_b) ** 2)

        if metric == 2 :
            a = region_a - region_a.mean()
            b = region_b - region_b.mean()
            den = np.sqrt(np.sum(a * a) * np.sum(b * b))
            if den < 1e-10: return 0
            return np.sum(a*b)/den

    def normalizate(img1, img2, shift):
        
        ### Args:
        #img1 = верхнее изображение
        #img2 = нижнее
        #двигаем верхнее относительно нижнего
        
        x, y = shift
        if y > 0 and x > 0:
            return img1[y : , x : ], img2[ : -y, : -x]
        
        if y < 0 and x > 0:
            return img1[:y, x:], img2[-y:, :-x]
        
        if x < 0 and y > 0:
            return img1[y:, :x], img2[:-y, -x:]
        
        if x < 0 and y < 0:
            return img1[:y, :x], img2[-y:, -x:]
        
        if y == 0 and x > 0:
            return img1[:, x:], img2[:, :-x]
        if y == 0 and x < 0:
            return img1[:, x:], img2[:, :-x]
        if x == 0 and y > 0:
            return img1[y:, :], img2[:-y, :]
        if x == 0 and y < 0:
            return img1[:y, :], img2[-y:, :]

        return img1,img2

    def best_shift(img1, img2, shift_range):
        
        ### Args:
        #img1 = верхнее изображение
        #img2 = нижнее
        #двигаем верхнее относительно нижнего
        
        bestSh = np.array([0,0])
        bestM2 = -np.inf
        for y in range (-shift_range, shift_range + 1):
            for x in range (-shift_range, shift_range + 1):
                
                if y >0 and x>0:
                    new_img1 = img1[y : , x : ]
                    new_img2 = img2[ : -y, : -x]
                elif y>0 and x<0:
                    new_img1 = img1[y:, :x]
                    new_img2 = img2[:-y, -x:]
                elif y<0 and x>0:
                    new_img1 = img1[:y, x:]
                    new_img2 = img2[-y:, :-x]
                elif y<0 and x<0:
                    new_img1 = img1[:y, :x]
                    new_img2 = img2[-y:, -x:]
                elif y == 0 and x>0:
                    new_img1 = img1[:, x:]
                    new_img2 = img2[:, :-x]
                elif y == 0 and x < 0:
                    new_img1 = img1[:, :x]
                    new_img2 = img2[:, -x:]
                elif x == 0 and y>0:
                    new_img1 = img1[y:]
                    new_img2 = img2[:-y]
                elif x == 0 and y<0:
                    new_img1 = img1[:y]
                    new_img2 = img2[-y:]
                else:
                    new_img1 = img1
                    new_img2 = img2


                #if x == -14 and y == -4:
                    #print(f"DEBUG shift [-14, 0]:")
                    #print(f"  new_img1 shape: {new_img1.shape}")
                    #print(f"  new_img2 shape: {new_img2.shape}")
                    #print(f"  new_img1 equals new_img2: {np.array_equal(new_img1, new_img2)}")
                
                    #metric2_debug = find_metric(new_img1, new_img2, 2)
                    #print(f"  metric: {metric2_debug}")
                
                ramki1 = new_img1.shape
                ramki2 = new_img2.shape

                itogy = min(ramki1[0], ramki2[0])
                itogx = min(ramki1[1], ramki2[1])

                new_img1 = new_img1[:itogy,:itogx]
                new_img2 = new_img2[:itogy, :itogx]

                metric2 = find_metric(new_img1, new_img2, 2)

                distance_penalty = 0.0001 * (abs(x) + abs(y)) 
                combined_metric = metric2 - distance_penalty

                if combined_metric > bestM2:
                    bestM2 = metric2
                    bestSh = np.array([x, y])

        

        return bestSh
    
    def mk_pyramid(img):
        pyramid = [img]
        new_img = img
        if min(new_img.shape) <= 600: return pyramid
        elif min(new_img.shape) <= 1000: stop = 500
        elif min(new_img.shape) >= 2000: stop = 70
        else: stop = 250
        
        while min(new_img.shape) > stop:
            #new_img = new_img[::2, ::2]
            h, w = new_img.shape
            if h % 2 == 1:
                new_img = new_img[:-1, :]
                h -= 1
            if w % 2 == 1:
                new_img = new_img[:, :-1]
                w -= 1

            #new_img = (new_img[::2, ::2] + new_img[1::2, ::2] + new_img[::2, 1::2] + new_img[1::2, 1::2]) / 4
            new_img = new_img.reshape(h//2, 2, w//2, 2).mean(axis=(1, 3))
            pyramid.append(new_img)
        return pyramid[::-1]
    
    pimg_a = mk_pyramid(img_a)
    pimg_b = mk_pyramid(img_b)

    #print("Pyramid levels:", len(pimg_a))
    #for i, img in enumerate(pimg_a):
        #print(f"Level {i}: {img.shape}")

    #print(img_a.shape)

    #print(f"Pyramid levels: {len(pimg_a)}")
    #print(f"Shapes: {[img.shape for img in pimg_a]}")

    best_sh = np.array([0,0])
    x= min(img_a.shape)
    if x <= 600: param = [15]
    elif x <= 1000:
        stop = 500
        param = [stop//16, stop//16//6, stop//16//6]
    elif x >= 2000:
        #stop = 70
        #param = [stop, stop//16, max(stop//16//6, 5)]
        #param = [90, 5, 5]
        param = [15, 8, 5, 3, 2, 1, 1, 1, 1]
    else: 
        stop =250
        param = [stop//16, stop//16//6, stop//16//6, stop//16//6, stop//16//6, stop//16//6]
        
    #param = [x, x//16, x//16//6]
    #bid_img = 0
    #if min(img_a.shape) > 2000:
        #bid_img = 2
   # else bid img
    #if bid_img:
        #best_sh = best_shift(pimg_a[0],pimg_b[0], 10)
    #else:
    best_sh = best_shift(pimg_a[0], pimg_b[0], param[0])
    #best_sh = best_sh(pimg_a[0], pimg_b[0], min(pimg_a[0]))
    #print(f"Level 0 shift: {best_sh}")

    prev_layerh = pimg_a[0].shape


    for i, (layer_a, layer_b) in enumerate(zip(pimg_a[1:],pimg_b[1:])):
        #print(f"Level before normalizate: best_sh = {best_sh}")
        current_range = param[i+1]
        #print(f"Level {i+1} shapes: {layer_a.shape}, prev_layerh: {prev_layerh}")

        k0 = (layer_a.shape[1] / prev_layerh[1])
        k1 = layer_a.shape[0] / prev_layerh[0]
        best_sh[0] = round(best_sh[0]*k0)
        best_sh[1] = round(best_sh[1]*k1)

        layer_a, layer_b = normalizate(layer_a, layer_b, best_sh)
        #print(f"Level after normalizate: shapes {layer_a.shape}, {layer_b.shape}")
        #if bid_img:
            #add_shift = best_shift(layer_a, layer_b, 20)
        #else:
        add_shift = best_shift(layer_a, layer_b, current_range)
        #print(f"Level additional_shift: {add_shift}")
        #print(f"Scaling factors: x = {layer_a.shape[1] / prev_layerh[1]}, y = {layer_a.shape[0] / prev_layerh[0]}")
        #print(f"Best shift before update: {best_sh}")

        best_sh[0] += add_shift[0]
        best_sh[1] += add_shift[1]
        
        prev_layerh = layer_a.shape
        #print(f"Level combined shift: {best_sh}, range={current_range}")

    #layer_a, layer_b = normalizate(pimg_a[-1], pimg_b[-1], best_sh)


    a_to_b = -(best_sh[::-1])
    #print(f"Final shift: {a_to_b}")
    return a_to_b


def find_absolute_shifts(
    crops,
    crop_coords,
    find_relative_shift_fn,
):
    red, green, blue = crops
    r_to_g_relative = find_relative_shift_fn(red, green)
    b_to_g_relative = find_relative_shift_fn(blue, green)

    #print(f"DEBUG: crop_coords = {crop_coords}")
    #print(f"DEBUG: relative shifts = {r_to_g_relative}, {b_to_g_relative}")

    r_to_g = -(crop_coords[0] - crop_coords[1] - r_to_g_relative)
    b_to_g = crop_coords[1] - crop_coords[2] + b_to_g_relative

    #print(f"DEBUG: r_to_g_abs = {r_to_g}")
    #print(f"DEBUG: b_to_g_abs = {b_to_g}")
    return r_to_g, b_to_g



def create_aligned_image(
    channels,
    channel_coords,
    r_to_g,
    b_to_g,
):
    r_to_g_rel = channel_coords[0] - channel_coords[1] + r_to_g
    b_to_g_rel = -(channel_coords[1] - channel_coords[2] - b_to_g)

    shifts = np.array([r_to_g_rel, [0,0], b_to_g_rel])

    red, green, blue = channels
    height, width = green.shape

    def bounds(y, x):
        top = max(0, y)
        left = max(0, x)
        bottom = min(height, height + y)
        right = min(width, width + x)
        return top, left, bottom, right

    g_top, g_left, g_bottom, g_right = 0, 0, height, width
    r_top, r_left, r_bottom, r_right = bounds(-r_to_g_rel[0], -r_to_g_rel[1])
    b_top, b_left, b_bottom, b_right = bounds(-b_to_g_rel[0], -b_to_g_rel[1])

    top = max(g_top, r_top, b_top)
    left = max(g_left, r_left, b_left)
    bottom = min(g_bottom, r_bottom, b_bottom)
    right = min(g_right, r_right, b_right)

    g_crop = green[top:bottom, left:right]
    r_crop = red[top + r_to_g_rel[0]:bottom + r_to_g_rel[0], left + r_to_g_rel[1]:right + r_to_g_rel[1]]
    b_crop = blue[top + b_to_g_rel[0]:bottom + b_to_g_rel[0], left + b_to_g_rel[1]:right + b_to_g_rel[1]]

    aligned_img = np.stack([r_crop, g_crop, b_crop], axis=-1)

    #print(f"DEBUG: green.dtype = {green.dtype}")
    '''
    aligned_img = np.full((height*3, width*3, 3), 20.0, dtype=green.dtype)
    aligned_img[height:2*height, width:2*width, 1] = green
    aligned_img[height + r_to_g_rel[0]: 2*height + r_to_g_rel[0], width + r_to_g_rel[1]: 2*width + r_to_g_rel[1], 0] = red
    aligned_img[height + b_to_g_rel[0]: 2*height + b_to_g_rel[0], width + b_to_g_rel[1]: 2*width + b_to_g_rel[1], 2] = blue
    mask0 = aligned_img[...,0] != 20
    mask1 = aligned_img[...,1] != 20
    mask2 = aligned_img[...,2] != 20
    mask = mask0 & mask1 & mask2
    
    rows = np.where(np.any(mask, axis=1))[0]
    cols = np.where(np.any(mask, axis=0))[0]

    aligned_img = aligned_img[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1, :]
    '''
    

    return aligned_img


def find_relative_shift_fourier(img_a, img_b):
    '''
    def smaller(img):
        h, w = img.shape
        new_img = img
        if h % 2 == 1:
            new_img = new_img[:-1, :]
        if w % 2 == 1:
            new_img = new_img[:, :-1]

        new_img = (new_img[::2, ::2] + new_img[1::2, ::2] + new_img[::2, 1::2] + new_img[1::2, 1::2]) / 4
        return new_img
    
    if min(img_a.shape) >=2000: 
        img_a = smaller(img_a)
        img_b = smaller(img_b)
    '''
    def apply_hann_window(img):
        h, w = img.shape
        window = np.outer(np.hanning(h), np.hanning(w))
        return img * window
    
    '''
    if max(img_a.shape) > 1000:
        img_a = img_a[::2, ::2]
        img_b = img_b[::2, ::2]
    '''
    if min(img_a.shape) >=2000: 
        img_a = apply_hann_window(img_a)
        img_b = apply_hann_window(img_b)


    img_a = np.fft.fft2(img_a)
    img_b = np.fft.fft2(img_b)

    R = img_a * np.conj(img_b)
    R /= (np.abs(img_a)*np.abs(np.conj(img_b))) + 1e-9

    cross_corr = np.fft.fftshift(np.real(np.fft.ifft2(R)))

    max_idx = np.unravel_index(np.argmax(cross_corr), img_a.shape)

    mid_y, mid_x = img_a.shape[0] // 2, img_b.shape[1] // 2
    a_to_b = -np.array([max_idx[0] - mid_y, max_idx[1] - mid_x])
    #a_to_b *= 2**2
    return a_to_b


if __name__ == "__main__":
    import common
    import pipeline

    # Read the source image and the corresponding ground truth information
    test_path = "tests/05_unittest_align_image_pyramid_img_small_input/00"
    raw_img, (r_point, g_point, b_point) = common.read_test_data(test_path)

    # Draw the same point on each channel in the original
    # raw image using the ground truth coordinates
    visualized_img = pipeline.visualize_point(raw_img, r_point, g_point, b_point)
    common.save_image(f"gt_visualized.png", visualized_img)

    for method in ["pyramid", "fourier"]:
        # Run the whole alignment pipeline
        r_to_g, b_to_g, aligned_img = pipeline.align_image(raw_img, method)
        common.save_image(f"{method}_aligned.png", aligned_img)

        # Draw the same point on each channel in the original
        # raw image using the predicted r->g and b->g shifts
        # (Compare with gt_visualized for debugging purposes)
        r_pred = g_point - r_to_g
        b_pred = g_point - b_to_g
        visualized_img = pipeline.visualize_point(raw_img, r_pred, g_point, b_pred)

        r_error = abs(r_pred - r_point)
        b_error = abs(b_pred - b_point)
        print(f"{method}: {r_error = }, {b_error = }")

        common.save_image(f"{method}_visualized.png", visualized_img)
