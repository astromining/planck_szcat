import numpy as np
def find_centroid(pic):
    from skimage.measure import moments
    import numpy as np
    
    if len(pic.shape) > 2:
        pic = np.copy(pic).reshape(list(pic.shape)[:-1])
    M = moments(pic)
    centroid = (M[1, 0] / M[0, 0], M[0, 1] / M[0, 0])
    
    return centroid

def pack_all_catalogs(cat_dir='/home/rt2122/Data/clusters/'):
    import os
    import numpy as np
    import pandas as pd
    
    all_cats = []
    files = next(os.walk(cat_dir))[-1]
    for file in files:
        df = pd.read_csv(os.path.join(cat_dir, file))
        df['catalog'] = file[:-4]
        all_cats.append(df)
    all_cats = pd.concat(all_cats, ignore_index=True)
    
    return all_cats

def get_radius(figure, center):
    import numpy as np
    from skimage.filters import roberts
    center = np.array(center)
    
    edge = np.where(roberts(figure) != 0)
    rads = []
    
    for point in zip(*edge):
        rads.append(np.linalg.norm(center - np.array(point)))
    if len(rads) == 0:
        return 0, 0, 0
    return min(rads), np.mean(rads), max(rads)

def divide_figures(pic):
    import numpy as np
    from skimage.segmentation import flood, flood_fill
    
    coords = np.array(np.where(pic != 0))
    ans = []
    while coords.shape[1] != 0:
        seed_point = tuple(coords[:, 0])
        ans.append(flood(pic, seed_point))
        pic = flood_fill(pic, seed_point, 0)
        
        coords = np.array(np.where(pic != 0))
    
    return ans

def find_centers_on_mask(mask, thr, binary=True):
    import numpy as np

    mask_binary = np.copy(mask)
    mask_binary = np.array(mask_binary >= thr, dtype=np.float32)
    
    figures = divide_figures(mask_binary)
    centers = []
    area = []
    min_rad = []
    mean_rad = []
    max_rad = []
    min_pred = []
    max_pred = []
    for figure in figures:
        f = np.zeros_like(mask)
        f[np.where(figure)] = mask[np.where(figure)]

        if not binary:
            centers.append(find_centroid(f))
        else:
            centers.append(find_centroid(figure))
        
        area.append(np.count_nonzero(figure))
        rads = get_radius(figure[:,:,0], centers[-1])
        min_rad.append(rads[0])
        mean_rad.append(rads[1])
        max_rad.append(rads[2])
        min_pred.append(np.partition(list(set(f.flatten())), 1)[1])
        max_pred.append(f.max())

    return {'centers' : np.array(centers), 'area' : np.array(area), 
            'min_rad' : np.array(min_rad), 'max_rad' : np.array(max_rad),
            'mean_rad' : np.array(mean_rad),
           'min_pred': np.array(min_pred), 'max_pred' : np.array(max_pred)}

def clusters_in_pix(clusters, pix, nside, search_nside=None):
    import pandas as pd
    import healpy as hp
    import numpy as np
    from DS_healpix_fragmentation import radec2pix
    
    df = pd.read_csv(clusters)
    cl_pix = radec2pix(df['RA'], df['DEC'], nside)
    df = df[cl_pix == pix]
    df.index = np.arange(df.shape[0])
    if not (search_nside is None):
        search_pix = radec2pix(df['RA'], df['DEC'], search_nside)
        df['search'] = search_pix
    
    return df

def gen_pics_for_detection(ipix, model, big_nside=2, step=64, size=64, depth=10, 
        mask_radius=15/60, clusters_dir='/home/rt2122/Data/clusters/', 
        planck_dirname='/home/rt2122/Data/Planck/normalized/', data_type=np.float64, 
        only=False, only_idx=100):
    from DS_healpix_fragmentation import one_pixel_fragmentation, pix2radec, radec2pix
    from DS_Planck_Unet import draw_pic_with_mask, draw_pic
    import pandas as pd
    import numpy as np
    import healpy as hp
    import os
    
    true_clusters = pack_all_catalogs(clusters_dir)
    clusters_pix = radec2pix(true_clusters['RA'], true_clusters['DEC'], 2)
    true_clusters = true_clusters[clusters_pix == ipix]
    true_clusters.index = np.arange(true_clusters.shape[0])
 
    big_matr = one_pixel_fragmentation(big_nside, ipix, depth)
    big_pic, big_mask = draw_pic_with_mask(center=None, matr=big_matr, 
                            mask_radius=mask_radius,
                            clusters_arr=np.array(true_clusters[['RA', 'DEC']]), 
                            dirname=planck_dirname)
    big_pic = big_pic.astype(data_type)
    
    pics, matrs, masks = [], [], []
    pic_idx = []

    starts = []
    for k in range(2):
        x_st = [i for i in range(0, big_matr.shape[k], step) 
                if i + size <= big_matr.shape[k]] + [big_matr.shape[k] - size]
        starts.append(x_st) 

    for i in starts[0]:
        for j in starts[1]:
            pic = big_pic[i:i+size,j:j+size,:]
            mask = big_mask[i:i+size,j:j+size,:]
            matr = big_matr[i:i+size,j:j+size]
            
            if pic.shape == (size, size, pic.shape[-1]):
                pics.append(pic)
                pic_idx.append((i, j))
                matrs.append(matr)
                masks.append(mask)
 
    ans = None
    if only:
        ans = []
        for i in range(0, len(pics), only_idx):
            ans.append(model.predict(np.array(pics[i:i+only_idx])))
        ans = np.concatenate(ans, axis=0)
    else:
        ans = model.predict(np.array(pics))
    if only:
        return {'ans' : ans, 'pic_idx' : pic_idx}
    return {'true_clusters' : true_clusters,
            'pics' : pics, 'matrs' : matrs, 'masks' : masks, 'ans' : ans,
            'pic_idx' : pic_idx} 

def detect_clusters_on_pic(ans, matr, thr, binary):
    import numpy as np
    dd = find_centers_on_mask(ans, thr, binary)
    if len(dd['centers']) > 0:
        centers = np.array(dd['centers'], dtype=np.int32)
        dd['centers'] = matr[centers[:,0], centers[:,1]]
    return dd


def connect_masks(ans, pic_idx, size=64, big_shape=(1024, 1024, 1), data_type=np.float64):
    import numpy as np
    
    connected_ans = np.zeros(big_shape, dtype=data_type)
    coef = np.zeros(big_shape, dtype=data_type)
    
    for i in range(len(ans)):
        x, y = pic_idx[i]
        connected_ans[x:x+size, y:y+size, :] += ans[i]
        coef[x:x+size,y:y+size, :] += np.ones((size, size, 1), dtype=data_type)
    
    connected_ans /= coef
    return connected_ans

def detect_clusters_connected(all_dict, thr, ipix, depth=10, 
                              base_nside=2048, tp_dist=5/60,
                             binary=False):
    import numpy as np
    import pandas as pd
    from DS_healpix_fragmentation import one_pixel_fragmentation, pix2radec
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    
    true_clusters = all_dict['true_clusters']
    big_ans = connect_masks(all_dict['ans'], all_dict['pic_idx'])
    big_matr = one_pixel_fragmentation(2, ipix, depth)
    
    dd = detect_clusters_on_pic(big_ans, big_matr, thr, binary)
    if len(dd['centers']) == 0:
        return pd.DataFrame([], index=[])
    
    res_cat = pd.DataFrame({'RA' : [], 'DEC' : [], 'area' : [], 
        'min_rad' : [], 'max_rad' : [], 'mean_rad':[],
                      'min_pred' : [], 'max_pred' : [], 
                      'tRA':[], 'tDEC' : [], 
                      'status' :[], 'catalog':[], 'M500' : [], 'z' : []})
    ra, dec = pix2radec(dd['centers'], nside=base_nside)
    res_cat['RA'] = ra
    res_cat['DEC'] = dec
    for prm in ['area', 'min_rad', 'max_rad', 'min_pred', 'max_pred', 'mean_rad']:
        res_cat[prm] = dd[prm]
    
    res_cat_sc = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
    true_clusters_sc = SkyCoord(ra=true_clusters['RA']*u.degree, 
                               dec=true_clusters['DEC']*u.degree)
    
    idx, d2d, _ = res_cat_sc.match_to_catalog_sky(true_clusters_sc)
    matched = d2d.degree <= tp_dist
    res_cat['status'] = 'fp'
    res_cat['status'].iloc[matched] = 'tp'
    res_cat['catalog'].iloc[matched] = np.array(
        true_clusters['catalog'][idx[matched]])
    res_cat['tRA'].iloc[matched] = np.array(true_clusters['RA'][idx[matched]])
    res_cat['tDEC'].iloc[matched] = np.array(true_clusters['DEC'][idx[matched]])
    res_cat['M500'].iloc[matched] = np.array(true_clusters['M500'][idx[matched]])
    res_cat['z'].iloc[matched] = np.array(true_clusters['z'][idx[matched]])
    
    res_cat_tp = res_cat[res_cat['status'] == 'tp']
    res_cat_tp = res_cat_tp.drop_duplicates(subset=['tRA', 'tDEC'])
    res_cat = pd.concat([res_cat[res_cat['status'] != 'tp'], res_cat_tp], 
                        ignore_index=True)
 
    
    true_clusters['found'] = False
    true_clusters['found'].iloc[idx[matched]] = True
    true_clusters['status'] = 'fn'
    true_clusters['tRA'] = true_clusters['RA']
    true_clusters['tDEC'] = true_clusters['DEC']
    
    res_cat = pd.concat([res_cat, true_clusters[['RA', 'DEC', 'status', 'catalog', 'M500', 'z', 
                            'tRA', 'tDEC']]
                         [true_clusters['found']==False]], ignore_index=True)
    return res_cat

def gen_catalog(models, big_pix, cat_name, step=8, thr=0.1, save_inter_cats=None, 
        clusters_dir='/home/rt2122/Data/clusters/', 
        planck_dirname='/home/rt2122/Data/Planck/normalized/', val_cats=None):
    from tqdm.notebook import tqdm
    import os
    import pandas as pd 
    from DS_Planck_Unet import load_planck_model, val_pix

    if not (val_cats is None):
        big_pix = list(set(big_pix) - set(val_pix))
    
    for model_name in tqdm(models):
        save_name = cat_name.format(model=model_name, thr=thr, step=step)
        if os.path.isfile(save_name):
            continue
        cur_cat = []
        for i in tqdm(big_pix):
            model = load_planck_model(models[model_name])
            all_dict = gen_pics_for_detection(i, model, step=step, clusters_dir=clusters_dir,
                    planck_dirname=planck_dirname)
            coords = detect_clusters_connected(all_dict, thr, i)
            cur_cat.append(coords)
            if not (save_inter_cats is None):
                coords.to_csv(save_inter_cats.format(pix=i, model=model_name), index=False)
        cur_cat = pd.concat(cur_cat, ignore_index=True)
        if not (val_cats is None):
            cur_cat = pd.concat([cur_cat, pd.read_csv(val_cats[model_name])])
        cur_cat.to_csv(save_name, index=False)


def rematch_cat(name, clusters_dir='/home/rt2122/Data/clusters/', tp_dist=5/60, add_fn=False):
    import numpy as np
    import pandas as pd
    import os
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    
    res_cat = pd.read_csv(name)
    true_clusters = pack_all_catalogs(clusters_dir)
    
    if 'status' in list(res_cat):
        res_cat = res_cat[res_cat['status'] != 'fn']
        res_cat.index = np.arange(len(res_cat))
    
    res_cat_sc = SkyCoord(ra=np.array(res_cat['RA'])*u.degree, 
                          dec=np.array(res_cat['DEC'])*u.degree, frame='icrs')
    true_clusters_sc = SkyCoord(ra=np.array(true_clusters['RA'])*u.degree, 
                               dec=np.array(true_clusters['DEC'])*u.degree)
    
    idx, d2d, _ = res_cat_sc.match_to_catalog_sky(true_clusters_sc)
    matched = d2d.degree <= tp_dist
    
    res_cat['status'] = 'fp'
    res_cat['status'].iloc[matched] = 'tp'
    res_cat['catalog'] = ''
    res_cat['catalog'].iloc[matched] = np.array(
        true_clusters['catalog'][idx[matched]])
    res_cat['tRA'] = np.nan
    res_cat['tRA'].iloc[matched] = np.array(true_clusters['RA'][idx[matched]])
    res_cat['tDEC'] = np.nan
    res_cat['tDEC'].iloc[matched] = np.array(true_clusters['DEC'][idx[matched]])
    if 'M500' in list(true_clusters):
        res_cat['M500'] = np.nan
        res_cat['M500'].iloc[matched] = np.array(true_clusters['M500'][idx[matched]])
    if 'z' in list(true_clusters):
        res_cat['z'] = np.nan
        res_cat['z'].iloc[matched] = np.array(true_clusters['z'][idx[matched]])
    if 'LAMBDA' in list(true_clusters):
        res_cat['LAMBDA'] = np.nan
        res_cat['LAMBDA'].iloc[matched] = np.array(true_clusters['LAMBDA'][idx[matched]])

    
    res_cat_tp = res_cat[res_cat['status'] == 'tp']
    res_cat_tp = res_cat_tp.drop_duplicates(subset=['tRA', 'tDEC'])
    res_cat = pd.concat([res_cat[res_cat['status'] != 'tp'], res_cat_tp], 
                        ignore_index=True)
 
    
    true_clusters['found'] = False
    true_clusters['found'].iloc[idx[matched]] = True
    true_clusters['status'] = 'fn'
    true_clusters['tRA'] = true_clusters['RA']
    true_clusters['tDEC'] = true_clusters['DEC']
    
    prm = ['RA', 'DEC', 'status', 'catalog', 'M500', 'z', 'tRA', 'tDEC', 'LAMBDA']
    prm = list(set(prm).intersection(list(true_clusters)))

    if add_fn:
        res_cat = pd.concat([res_cat, true_clusters[prm][true_clusters['found']==False]], 
            ignore_index=True)
    return res_cat
