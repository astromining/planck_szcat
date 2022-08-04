val_pix = [9, 38, 41]
test_pix = [6]
train_pix = [i for i in range(48) if not (i in val_pix) and not (i in test_pix)]
train_pix_act = [2, 8, 10, 11, 13, 15, 32, 33, 36, 37, 40, 42, 43, 44, 45, 46]
planck_side = 2048
min_rad_64 = 0.62
east_val = [38, 6]
west_val = [9, 41]

east = [0,  1,  2,  3,  4,  5,  6,  7, 16, 17, 20, 21, 22, 23, 24, 26, 27,
       32, 33, 34, 35, 36, 37, 38, 39]
west = [ 8,  9, 10, 11, 12, 13, 14, 15, 18, 19, 25, 28, 29, 30, 31, 40, 41,
       42, 43, 44, 45, 46, 47]

def gen_matr(ra, dec, radius=0.84, size=64, fin_nside=2048):
    from DS_healpix_fragmentation import find_biggest_pixel, one_pixel_fragmentation,\
        draw_proper_circle, matr2dict
    import numpy as np
    

    big_nside, big_ipix = find_biggest_pixel(ra, dec, radius)
    depth = int(np.log2(fin_nside // big_nside))
    big_matr = one_pixel_fragmentation(big_nside, big_ipix, depth)
    big_dict = matr2dict(big_matr)
    
    
    circle_coords = draw_proper_circle(ra, dec, radius, fin_nside, big_dict, 
                                       big_matr.shape, coords_mode=True)
    mins = [None, None]
    maxs = [None, None]
    for i in range(2):
        dif = circle_coords[:,i].min() - circle_coords[:,i].max() + size
        mins[i] = circle_coords[:,i].min() - dif
        maxs[i] = circle_coords[:,i].max() + dif
        if maxs[i] - mins[i] != size:
            maxs[i] -= maxs[i] - mins[i] - size
    
   
    return big_matr[mins[0]:maxs[0],mins[1]:maxs[1]]

def draw_pic(matr, dirname='/home/rt2122/Data/Planck/normalized', y=False):
    import os
    import numpy as np
   
    if y:
        dirname = os.path.join(dirname, 'y')
        
    files = sorted(next(os.walk(dirname))[-1])
    pic = np.zeros(list(matr.shape) + [len(files)])
    
    for i_f, file in enumerate(files):
        i_s = np.load(os.path.join(dirname, file))
        
        for x in range(pic.shape[0]):
            pic[x, :, i_f] = i_s[matr[x]]
    return pic

def draw_pic_with_mask(center, clusters_arr, radius=0.84, size=64, fin_nside=2048, 
                       dirname='/home/rt2122/Data/Planck/normalized/', 
                      mask_radius=2.5/60, retmatr=False, matr=None, centers_in_patch=False):
    from DS_healpix_fragmentation import matr2dict, draw_proper_circle
    import numpy as np
    
    if matr is None:
        matr = gen_matr(center[0], center[1], radius, size, fin_nside)
        
    mdict = matr2dict(matr)
    
    pic = draw_pic(matr, dirname)
    mask = np.zeros(list(matr.shape) + [1], dtype=np.uint8)
    for ra, dec in clusters_arr:
        mask = np.logical_or(mask, 
            draw_proper_circle(ra, dec, mask_radius, fin_nside, mdict, 
                              mask.shape, coords_mode=False, centers_in_patch=centers_in_patch))
    if not retmatr:
        return pic, mask

    return pic, mask, matr

def nearest_clusters(df, theta, phi, radius=2, galactic=True):
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    import numpy as np
    sc_cen = None
    sc_cl = SkyCoord(ra=np.array(df['RA'])*u.degree, 
                     dec=np.array(df['DEC'])*u.degree, frame='icrs')
    if galactic:
        sc_cen = SkyCoord(l=theta*u.degree, b=phi*u.degree, frame='galactic')
    else:
        sc_cen = SkyCoord(ra=theta*u.degree, dec=phi*u.degree, frame='icrs')
    return df[sc_cen.separation(sc_cl).degree < radius]

def pixels_with_clusters(clusters_files, big_pixels, nside, min_rad=0.62):
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    import pandas as pd
    import healpy as hp
    import numpy as np
    from DS_healpix_fragmentation import radec2pix

    
    df = [pd.read_csv(clusters_file) for clusters_file in clusters_files]
    df = pd.concat(df, ignore_index=True)

    pix2 = radec2pix(df['RA'], df['DEC'], 2)
    df = df[np.in1d(pix2, big_pixels)]
    df.index = np.arange(df.shape[0])
    
    small_pixels = set()
    sc = SkyCoord(ra=np.array(df['RA'])*u.degree, 
                  dec=np.array(df['DEC'])*u.degree, frame='icrs')
    vecs = hp.ang2vec(theta=sc.galactic.l.degree, phi=sc.galactic.b.degree, 
                      lonlat=True)
    for i in range(df.shape[0]):
        small_pixels = small_pixels.union(hp.query_disc(nside, vecs[i], 
                                                np.radians(min_rad), nest=True))
    small_pixels = np.array(list(small_pixels))
    return small_pixels, df

def gen_batch(pixels_of_choice, batch_size, nside_choice, clusters, retmatr=False, size=64,
        print_coords=False, centers_in_patch=False):
    import numpy as np
    import healpy as hp
    from astropy.coordinates import SkyCoord
    from astropy import units as u

    ipix = np.random.choice(pixels_of_choice, batch_size)
    theta, phi = hp.pix2ang(nside=nside_choice, nest=True, ipix=ipix, lonlat=True)

    sc = SkyCoord(l=theta*u.degree, b=phi*u.degree, frame='galactic')
    ra = sc.icrs.ra.degree
    dec = sc.icrs.dec.degree
    pics = []
    matrs = []
    masks = []

    i = 0
    while i < batch_size:
        cl_list = nearest_clusters(clusters, theta[i], phi[i], galactic=True)
        cl_list = np.stack([cl_list['RA'], cl_list['DEC']]).T
        pic, mask = None, None
        matr = None
        ret = draw_pic_with_mask([ra[i], dec[i]], cl_list, retmatr=retmatr,
                centers_in_patch=centers_in_patch)
        pic = ret[0]
        mask = ret[1]
        if retmatr:
            matr = ret[2]

        if not(pic.shape[0] == size and pic.shape[1] == size) or np.count_nonzero(mask) == 0:
            pixels_of_choice= pixels_of_choice[pixels_of_choice != ipix[i]]
            ipix[i] = np.random.choice(pixels_of_choice)
            theta[i], phi[i] = hp.pix2ang(nside=nside_choice, nest=True, 
                    ipix=ipix[i], lonlat=True)
            sc_cur = SkyCoord(l=theta[i]*u.degree, b=phi[i]*u.degree, frame='galactic')
            ra[i] = sc_cur.icrs.ra.degree
            dec[i] = sc_cur.icrs.dec.degree
        else:
            pics.append(pic)
            matrs.append(matr)
            masks.append(mask)
            if print_coords:
                print(ra[i], dec[i])
            i += 1

    if retmatr:
        return pics, masks, matrs

    return pics, masks


def gen_data(clusters_files, big_pixels, batch_size, nside=2048, min_rad=0.08, search_nside=512,
        size=64, retmatr=False, print_coords=False, centers_in_patch=False):
    import healpy as hp
    import numpy as np
    from tensorflow import convert_to_tensor
    
    small_pixels, df = pixels_with_clusters(clusters_files, big_pixels, search_nside, min_rad)
    
    while True:
        if retmatr:
            pics, masks, matrs = gen_batch(small_pixels, batch_size, search_nside, df, size=size,
                    retmatr=retmatr, print_coords=print_coords, centers_in_patch=centers_in_patch)
            yield np.stack(pics), np.stack(masks) , np.stack(matrs)
        else:
            pics, masks = gen_batch(small_pixels, batch_size, search_nside, df, size=size,
                    retmatr=retmatr, print_coords=print_coords, centers_in_patch=centers_in_patch)
            yield convert_to_tensor(np.stack(pics)), convert_to_tensor(np.stack(masks)) 



def iou(y_pred, y_true):
    from tensorflow.keras import backend as K
    iou_sum = 0
    for i in range(y_true.shape[-1]):
        inters = K.sum(y_pred[..., i] * y_true[..., i])
        union = K.sum((y_pred[..., i] + y_true[..., i])) - inters
        iou_sum += inters / union
    return iou_sum

def dice(y_pred, y_true, eps=0.1):
    from tensorflow.keras import backend as K
    dice_sum = 0
    for i in range(y_true.shape[-1]):
        inters = K.sum(y_pred[..., i] * y_true[..., i])
        union = K.sum((y_pred[..., i] + y_true[..., i])) - inters
        dice_sum += K.mean((2 * inters + eps) / (union + eps))
    return dice_sum

def unet_planck(input_size = (64,64,6), filters=8, blocks=5, output_layers=1, weights=None, 
        lr=1e-4, add_batch_norm=False, dropout_prm=0.2): 
    #import numpy as np 
    from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
    from tensorflow.keras.optimizers import Adam, SGD
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, concatenate, UpSampling2D
    from tensorflow.keras import Input
    from tensorflow.keras.callbacks import ModelCheckpoint
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Activation
    from tensorflow.keras.activations import relu, sigmoid, softmax
    from tensorflow.keras.layers import UpSampling2D
    from tensorflow.keras.losses import binary_crossentropy, categorical_crossentropy,\
            sparse_categorical_crossentropy
    from tensorflow.keras.layers import BatchNormalization, Dropout
    from tensorflow.keras import backend as K
    from tensorflow.keras.models import load_model

    encoder = []
    inputs = Input(input_size)
    prev = inputs
    for i in range(blocks):
        cur = Conv2D(filters=filters, kernel_size=(3, 3), padding = 'same', 
                kernel_initializer = 'he_normal')(prev)
        if add_batch_norm:
            cur = BatchNormalization()(cur)
        else:
            cur = Dropout(dropout_prm)(cur)
        cur = Activation(relu)(cur)

        cur = Conv2D(filters=filters, kernel_size=(3, 3), padding = 'same', 
                kernel_initializer = 'he_normal')(cur)

        if add_batch_norm:
            cur = BatchNormalization()(cur)
        else:
            cur = Dropout(dropout_prm)(cur)
        cur = Activation(relu)(cur)

        encoder.append(cur)

        cur = MaxPooling2D(padding='valid')(cur)

        filters *= 2
        prev = cur
    for i in range(blocks - 1, -1, -1):
        cur = UpSampling2D()(prev)
        cur = Conv2D(filters=filters, kernel_size=3, padding='same')(cur)
        if not add_batch_norm:
            cur = Dropout(dropout_prm)(cur)
        cur = Activation(relu)(cur)
        cur = concatenate([cur, encoder[i]], axis=3)

        cur = Conv2D(filters=filters, kernel_size=3, padding='same')(cur)
        cur = Activation(relu)(cur)
        if not add_batch_norm:
            cur = Dropout(dropout_prm)(cur)
        #cur = Conv2D(filters=filters, kernel_size=3, padding='same')(cur)
        #cur = Activation(relu)(cur)

        prev = cur
        filters //= 2


    prev = Conv2D(output_layers, kernel_size=3, padding='same')(prev)
    prev = Activation(sigmoid)(prev)

    if not (weights is None):
        pt = Model(inputs=inputs, outputs=prev)
        pt.load_weights(weights)
        pt.compile()
        return pt
    
    model = Model(inputs=inputs, outputs=prev)
    model.compile(optimizer = Adam(lr = lr), loss = binary_crossentropy, 
            metrics = ['accuracy', iou, dice])
    
    return model

def load_planck_model(path):
    from tensorflow.keras.models import load_model
    return load_model(path, custom_objects={'iou' : iou, 'dice' : dice})

def check_gen(gen, model=None, thr=0.8, y=False):
    from matplotlib import pyplot as plt
    import numpy as np
    pic, mask, matr = None, None, None
    if y:
        pic, mask, matr = next(gen)
    else:
        pic, mask = next(gen)
    print(pic.shape, mask.shape)
    pic = pic[0]
    mask = mask[0]
    fig, ax = plt.subplots(3, 3, figsize=(12, 12))
    for i in range(3):
        for j in range(2):
            ax[i][j].imshow(pic[:,:,i+3*j])
    ax[0][2].imshow(mask[:,:,0])
    if not (model is None):
        ans = model.predict(np.array([pic]))
        ax[1][2].imshow(ans[0,:,:,0])
        if y:
            cpic = draw_pic(matr[0], y=True)
            ax[2][2].imshow(cpic[:,:,0])
        else:
            ax[2][2].imshow((ans[0,:,:,0] >= thr).astype(np.float32))

def check_mask(gen, model, thr_list):
    from matplotlib import pyplot as plt
    import numpy as np


    pic, mask = next(gen)
    ans = model.predict(pic)
    pic = pic[0]
    mask = mask[0]
    fig, ax = plt.subplots(3, len(thr_list) // 3 + 1, figsize=(12, 12))
    for i in range(3):
        for j in range(len(thr_list) // 3 + 1):
            if i + 3*j < len(thr_list):
                ax[i][j].imshow((ans[0,:,:,0] >= thr_list[i + 3*j]).astype(np.float32))
                ax[i][j].set_xlabel(thr_list[i+3*j])
    ax[-1][-1].imshow(mask[:,:,0])

def gen_data_from_pregen(path, batch_size):
    import os
    import numpy as np
    from tensorflow import convert_to_tensor
    
    pic_names = next(os.walk(os.path.join(path, 'x')))[-1]
    n_pics = len(pic_names)
    while True:
        idx_sh = np.arange(n_pics)
        np.random.shuffle(idx_sh)
        for st in range(0, n_pics, batch_size):
            x = []
            y = []
            for idx in idx_sh[st:st+batch_size]:
                #print('#', pic_names[idx], '#')
                x.append(np.load(os.path.join(path, 'x', pic_names[idx])))
                y.append(np.load(os.path.join(path, 'y', pic_names[idx])))
            #yield convert_to_tensor(np.stack(x).astype(np.float64)), \
            #        convert_to_tensor(np.stack(y).astype(np.float64))
            yield np.stack(x).astype(np.float64), \
                    np.stack(y).astype(np.float64)
def gen_data_while_training(big_pixels, batch_size, patches_file, pregen_pics, pregen_masks, size=64, 
        no_sample=False, rotate_aug=False):
    import numpy as np
    import pandas as pd
    import os
    
    patches = pd.read_csv(patches_file)
    patches = patches[np.in1d(patches['pix'], big_pixels)]
    patches.index = np.arange(len(patches))
    
    pic_dict = {}
    mask_dict = {}
    
    for pix in big_pixels:
        pic_dict[pix] = np.load(os.path.join(pregen_pics, str(pix) + '.npy'))
        mask_dict[pix] = np.load(os.path.join(pregen_masks, str(pix) + '.npy'))
    
    while True:
        sample = patches.sample(frac=1)
        if no_sample:
            sample = patches
        
        for i in range(0, len(patches) - batch_size, batch_size):
            pics = []
            masks = []
            
            cur_pix = np.array(sample['pix'].iloc[i:i+batch_size])
            cur_x = np.array(sample['x'].iloc[i:i+batch_size])
            cur_y = np.array(sample['y'].iloc[i:i+batch_size])
            
            for j in range(batch_size):
                pix = cur_pix[j]
                x = cur_x[j]
                y = cur_y[j]
                
                pics.append(pic_dict[pix][x:x+size,y:y+size,:])
                masks.append(mask_dict[pix][x:x+size,y:y+size,:])
                
                if rotate_aug:
                    n_rot = np.random.randint(4)
                    pics[-1] = np.rot90(pics[-1], n_rot)
                    masks[-1] = np.rot90(masks[-1], n_rot)
                
            
            yield np.stack(pics), np.stack(masks)
