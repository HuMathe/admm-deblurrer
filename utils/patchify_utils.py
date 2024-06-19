import numpy as np
import itertools
from scipy import sparse
from scipy.sparse.linalg import spsolve

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

def pathified_map(
        fn,
        image,
        pad_size,
        patch_size=(256, 256)):
    
    stich_type = {}
    stitches = {}

    def _count_cache():
        cache = {}
        inv_val = {}
        def inner(obj):
            if obj in cache:
                return cache[obj]
            idx = len(cache)
            cache[obj] = idx
            inv_val[idx] = obj
            return idx
        
        def inner_inv(idx):
            return inv_val[idx]
        return inner, inner_inv

    def _add_stiches(i, j, len_h, len_w, loc, data):
        samples:list
        samples = stitches.get((i, j), [])
        assert data.shape == (len_h, len_w)

        samples.append(data)

        stitches[(i, j)] = samples
        stich_type[(i, j)] = loc

    image_h, image_w = image.shape[:2]
    pad_h, pad_w = pad_size
    patch_h, patch_w = patch_size

    ret_image = np.zeros((image.shape[0] - pad_h*2, image.shape[1] - pad_w*2), image.dtype)
    pbar = itertools.product(range(pad_h, image_h - pad_h, patch_h), range(pad_w, image_w - pad_w, patch_w))
    if tqdm is not None:
        pbar = tqdm(list(pbar))
    for i, j in pbar:
        padded_image = image[i-pad_h:i+patch_h+pad_h,j-pad_w:j+patch_w+pad_w]

        local_ret = fn(padded_image)


        ret_image[i-pad_h:i-pad_h+patch_h, j-pad_w:j-pad_w+patch_w] = local_ret[pad_h:-pad_h,pad_w:-pad_w]

        rph, rpw = local_ret.shape[0] - 2*pad_h, local_ret.shape[1] - 2*pad_w
        stitch_regions = [
            [i-2*pad_h,     j-2*pad_w,      2*pad_h, 2*pad_w, 'c'],
            [i-2*pad_h+rph, j-2*pad_w,      2*pad_h, 2*pad_w, 'c'],
            [i-2*pad_h,     j-2*pad_w+rpw,  2*pad_h, 2*pad_w, 'c'],
            [i-2*pad_h+rph, j-2*pad_w+rpw,  2*pad_h, 2*pad_w, 'c'],
            
            [i-1,           j-2*pad_w,      rph-2*pad_h+2,  2*pad_w,        'ev'],
            [i-2*pad_h,     j-1,            2*pad_h,        rpw-2*pad_w+2,  'eh'],
            [i-1,           j-2*pad_w+rpw,  rph-2*pad_h+2,  2*pad_w,        'ev'],
            [i-2*pad_h+rph, j-1,            2*pad_h,        rpw-2*pad_w+2,  'eh']
        ]
        for (si, sj, li, lj, pos) in stitch_regions:
            ei, ej = si+li, sj+lj
            si, sj = max(si, 0), max(sj, 0)
            ei, ej = min(ei, ret_image.shape[0]), min(ej, ret_image.shape[1])
            li, lj = ei - si, ej - sj
            data = local_ret[si-i+2*pad_h:ei-i+2*pad_h, sj-j+2*pad_w:ej-j+2*pad_w]
            _add_stiches(si, sj, li, lj, pos, data)

    overlap_patches = sorted([
        [k, len(data), np.sum(data, axis=0) / len(data)]
        for k, data in stitches.items() if len(data) > 1],
        key=lambda x: x[1])

    for (min_i, min_j), otimes, avg_data in overlap_patches:
        max_i, max_j = min_i + avg_data.shape[0], min_j + avg_data.shape[1]
        row, col, val = [], [], []
        ind_f, inv_f = _count_cache()
        b = np.zeros(avg_data.shape[0] * avg_data.shape[1])

        if otimes == 2:
            if min_i == 0 or max_i == ret_image.shape[0] or stich_type[(min_i, min_j)] == 'eh':
                for i in range(min_i+1,max_i-1):
                    row.extend(map(ind_f, [(i, min_j), (i, min_j), (i, min_j), (i, min_j)]))
                    col.extend(map(ind_f, [(i, min_j), (i-1, min_j), (i+1, min_j), (i, min_j+1)]))
                    val.extend([3, -1, -1, -1])
                    b[ind_f((i, min_j))] = 3 * avg_data[i-min_i, 0] \
                                           - avg_data[i-min_i, 1] \
                                           - avg_data[i-min_i-1, 0] \
                                           - avg_data[i-min_i+1, 0]
                    
                    row.extend(map(ind_f, [(i, max_j-1), (i, max_j-1), (i, max_j-1), (i, max_j-1)]))
                    col.extend(map(ind_f, [(i, max_j-1), (i-1, max_j-1), (i+1, max_j-1), (i, max_j-2)]))
                    val.extend([3, -1, -1, -1])
                    b[ind_f((i, max_j-1))] = 3 * avg_data[i-min_i, -1] \
                                           - avg_data[i-min_i, -2] \
                                           - avg_data[i-min_i-1, -1] \
                                           - avg_data[i-min_i+1, -1]
                    

                    
                    for j in range(min_j+1, max_j-1):
                        row.extend(map(ind_f, [(i, j), (i, j), (i, j), (i, j), (i, j)]))
                        col.extend(map(ind_f, [(i, j), (i, j-1), (i, j+1), (i+1, j), (i-1, j)]))
                        val.extend([4, -1, -1, -1, -1])
                        b[ind_f((i, j))] = avg_data[i-min_i,j-min_j]*4 \
                                           - avg_data[i-min_i,j-min_j-1] \
                                           - avg_data[i-min_i,j-min_j+1] \
                                           - avg_data[i-min_i+1,j-min_j] \
                                           - avg_data[i-min_i-1,j-min_j]
                        
                for j in range(min_j, max_j):
                    row.extend(map(ind_f, [(min_i, j), (max_i-1, j)]))
                    col.extend(map(ind_f, [(min_i, j), (max_i-1, j)]))
                    val.extend([1, 1])
                    b[ind_f((min_i, j))] = ret_image[min_i, j]
                    b[ind_f((max_i-1, j))] = ret_image[max_i-1, j]
                    
    
            if min_j == 0 or max_j == ret_image.shape[1] or stich_type[(min_i, min_j)] == 'ev':
                for j in range(min_j+1, max_j-1):
                    row.extend(map(ind_f, [(min_i, j), (min_i, j), (min_i, j), (min_i, j)]))
                    col.extend(map(ind_f, [(min_i, j), (min_i, j-1), (min_i, j+1), (min_i+1, j)]))
                    val.extend([3, -1, -1, -1])
                    b[ind_f((min_i, j))] = avg_data[0, j-min_j] * 3 - avg_data[0, j-min_j-1] - avg_data[0, j-min_j+1] - avg_data[1, j-min_j]

                    row.extend(map(ind_f, [(max_i-1, j), (max_i-1, j), (max_i-1, j), (max_i-1, j)]))
                    col.extend(map(ind_f, [(max_i-1, j), (max_i-1, j-1), (max_i-1, j+1), (max_i-2, j)]))
                    val.extend([3, -1, -1, -1])
                    b[ind_f((max_i-1, j))] = avg_data[-1, j-min_j] * 3 - avg_data[-1, j-min_j-1] - avg_data[-1, j-min_j+1] - avg_data[-2, j-min_j]

                    for i in range(min_i+1, max_i-1):
                        row.extend(map(ind_f, [(i, j), (i, j), (i, j), (i, j), (i, j)]))
                        col.extend(map(ind_f, [(i, j), (i, j-1), (i, j+1), (i+1, j), (i-1, j)]))
                        val.extend([4, -1, -1, -1, -1])
                        b[ind_f((i, j))] = avg_data[i-min_i,j-min_j]*4 \
                                           - avg_data[i-min_i,j-min_j-1] \
                                           - avg_data[i-min_i,j-min_j+1] \
                                           - avg_data[i-min_i+1,j-min_j] \
                                           - avg_data[i-min_i-1,j-min_j]
                for i in range(min_i, max_i):
                    row.extend(map(ind_f, [(i, min_j), (i, max_j-1)]))
                    col.extend(map(ind_f, [(i, min_j), (i, max_j-1)]))
                    val.extend([1, 1])
                    b[ind_f((i, min_j))] = ret_image[i, min_j]
                    b[ind_f((i, max_j-1))] = ret_image[i, max_j-1]
                
        elif otimes == 4:
            for i in range(min_i+1, max_i-1):
                for j in range(min_j+1, max_j-1):
                    row.extend(map(ind_f, [(i, j), (i, j), (i, j), (i, j), (i, j)]))
                    col.extend(map(ind_f, [(i, j), (i, j-1), (i, j+1), (i+1, j), (i-1, j)]))
                    val.extend([4, -1, -1, -1, -1])
                    b[ind_f((i, j))] = avg_data[i-min_i,j-min_j]*4 \
                                        - avg_data[i-min_i,j-min_j-1] \
                                        - avg_data[i-min_i,j-min_j+1] \
                                        - avg_data[i-min_i+1,j-min_j] \
                                        - avg_data[i-min_i-1,j-min_j]
            for i in range(min_i, max_i):
                row.extend(map(ind_f, [(i, min_j), (i, max_j-1)]))
                col.extend(map(ind_f, [(i, min_j), (i, max_j-1)]))
                val.extend([1, 1])
                b[ind_f((i, min_j))] = ret_image[i, min_j]
                b[ind_f((i, max_j-1))] = ret_image[i, max_j-1]
            
            for j in range(min_j, max_j):
                    row.extend(map(ind_f, [(min_i, j), (max_i-1, j)]))
                    col.extend(map(ind_f, [(min_i, j), (max_i-1, j)]))
                    val.extend([1, 1])
                    b[ind_f((min_i, j))] = ret_image[min_i, j]
                    b[ind_f((max_i-1, j))] = ret_image[max_i-1, j]
        
        A = sparse.csr_matrix((val, (row, col)))
        x = spsolve(A, b)
        for idx in range(len(x)):
            ret_image[inv_f(idx)] = 0.2 * x[idx] + 0.8 * ret_image[inv_f(idx)]

    return np.clip(ret_image, 0, 1)

