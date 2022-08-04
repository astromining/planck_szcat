def fits2df(fitsname, filtered=False, one_col=None):
    from astropy.io import fits
    from astropy.table import Table
    import numpy as np
    
    df = None
    with fits.open(fitsname) as hdul:
        if not (one_col is None):
            return np.array(hdul[1].data[one_col])
        tbl = Table(hdul[1].data)
        names = [name for name in tbl.colnames if len(tbl[name].shape) <= 1]
        df = tbl[names].to_pandas()
    if filtered:
        df['phot_is_star_gaia'] = 0 
        sn2 = 4**2
        f = ~((df['PARALLAX'].isna()) | (df['PARALLAX']==0))
        df.loc[f,'phot_is_star_gaia'] = df.loc[f,'phot_is_star_gaia'] + \
            ((df.loc[f,'PARALLAX']**2) * df.loc[f,'PARALLAX_IVAR'] > sn2).astype(int)
        f = ~((df['PMRA'].isna()) | (df['PMRA']==0))
        df.loc[f,'phot_is_star_gaia'] = df.loc[f,'phot_is_star_gaia'] + \
            ((df.loc[f,'PMRA']**2) * df.loc[f,'PMRA_IVAR'] > sn2).astype(int)
        f = ~((df['PMDEC'].isna()) | (df['PMDEC']==0))
        df.loc[f,'phot_is_star_gaia'] = df.loc[f,'phot_is_star_gaia'] + \
            ((df.loc[f,'PMDEC']**2) * df.loc[f,'PMDEC_IVAR'] > sn2).astype(int)
        df = df[df['phot_is_star_gaia']==0]
        df.index = np.arange(df.shape[0])
    return df

def pic2fits(pic, wcs, fitsname):
    from astropy.io import fits
    import numpy as np

    hdul = None
    if wcs is None:
        hdul = fits.HDUList([fits.PrimaryHDU(), 
            fits.ImageHDU(np.stack([pic[:,:,i] for i in range(pic.shape[-1])]))]) 
    else:       
        hdul = fits.HDUList([fits.PrimaryHDU(), 
            fits.ImageHDU(np.stack([pic[:,:,i] for i in range(pic.shape[-1])]), 
                         header=wcs.to_header())])

    hdul.writeto(fitsname)

def show_pic(pic, projection=None, label = 'label', figsize=(10, 10), vmin=0, vmax=1, 
        slices=None):
    from matplotlib import pyplot as plt
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.1,0.1,0.8,0.8], projection=projection)
    plt.xlabel(label)
    if not (projection is None):
        ra = ax.coords[0]
        ra.set_format_unit('degree')

    im = ax.imshow(pic, cmap=plt.get_cmap('viridis'), 
                   interpolation='nearest', vmin=0, vmax=1)


def nth_max(array, n):
    import numpy as np
    return np.partition(array, -n)[-n]

def n_max_flux(flux, n):
    import numpy as np
    max_n = nth_max(np.array(flux), n)
    return flux >= max_n

def n_max_flux_df(df, n, ch):
    import numpy as np
    if type(ch) == type(''):
        ch = df[ch]
    else:
        ch = df[ch].sum(axis=1)
    df = df[n_max_flux(ch, n)]
    df.index = np.arange(df.shape[0])
    return df

def normalize(pic):
    import numpy as np

    if len(pic.shape) == 2:
        pic -= np.mean(pic)
        pic /= np.std(pic)
        return pic

    pic = np.copy(pic)
    for i in range(pic.shape[-1]):
        pic[:,:,i] -= np.mean(pic[:,:,i])
        pic[:,:,i] /= np.std(pic[:,:,i])
    return pic


def draw_df(data, base, figsize=(8, 6), grids=True, xgrid=None, ygrid=None,
               save=False, invert=True, comment='', comm_coords=(0.6, 0.1)):
    from matplotlib import pyplot as plt
    import numpy as np
    fig, ax = plt.subplots(1, figsize=figsize)
    colors = 'bgrcmykw'
    if invert:
        plt.gca().invert_xaxis()
    max_y = 0
    min_y = 100
    for c, label in zip(colors[:len(data)], data):
        line, = ax.plot(base, data[label], c+'o-')
        line.set_label(label)
        max_y = max(max_y, max(data[label]))
        min_y = min(min_y, min(data[label]))
    ax.legend()
    if grids:
        if xgrid is None:
            ax.set_xticks(base)
        else:
            ax.set_xticks(xgrid)
            
        if ygrid is None:
            ax.set_yticks(np.arange(min_y, max_y, (max_y-min_y) / 10))
        else:
            ax.set_yticks(ygrid)
        #ax.grid(True)
        plt.grid(b=True, which='major', color='#666666', linestyle=':')
    plt.text(*comm_coords, comment)
    if save:
        plt.savefig('pic.png')
    plt.show()


def get_prm(prm, s, w='\d'):
    import re
    return re.search(prm + r'(' + w + '+)', s)[0][len(prm):]

def calc_error(det_cat, true_cat, shift=15/60, match_dist=5/60, n_try=20, seed=0, sc_mode=False):
    import numpy as np
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    
    error = []
    np.random.seed(seed)
    true_sc = true_cat
    det_sc = det_cat
    if not sc_mode:
        true_sc = SkyCoord(ra=np.array(true_cat['RA']) * u.degree, 
                       dec=np.array(true_cat['DEC']) * u.degree, frame='icrs')
        det_sc = SkyCoord(ra=np.array(det_cat['RA']) * u.degree, 
                          dec=np.array(det_cat['DEC']) * u.degree, frame='icrs')
    for _ in range(n_try):
        angles = np.random.randint(0, 360, len(det_cat))
        det_sc_off = det_sc.directional_offset_by(angles*u.degree, shift)

        _, d2d, _ = det_sc_off.match_to_catalog_sky(true_sc)
        c_error = np.count_nonzero(d2d.degree < match_dist)
        error.append(c_error)
    error = np.array(error)
    return error.mean(), error.std() / np.sqrt(n_try - 1)

def extract_stats_from_df(df_name):
    import numpy as np
    import pandas as pd
    
    df = pd.read_csv(df_name)
    
    stat_df = {}
    stat_df['all'] = len(df)
    stat_df['all_tp'] = np.count_nonzero(df['status'] == 'tp')
    stat_df['all_fn'] = np.count_nonzero(df['status'] == 'fn')
    stat_df['fp'] = np.count_nonzero(df['status'] == 'fp')
    
    for cat in set(df['catalog']):
        if not (type(cat) == type('')):
            continue
        
        cur_df = df[df['catalog'] == cat]
        cur_df.index = np.arange(len(cur_df))
            
        stat_df[cat+'_tp'] = np.count_nonzero(cur_df['status'] == 'tp')
        stat_df[cat+'_fn'] = np.count_nonzero(cur_df['status'] == 'fn')
        stat_df[cat+'_recall'] = stat_df[cat+'_tp'] / len(cur_df)
    
    return stat_df

def stats_by_epoch(dir_name):
    import os
    import pandas as pd
    import numpy as np
    from DS_data_transformation import get_prm
    
    res_df = []
    
    files = next(os.walk(dir_name))[-1]
    for file in files:
        stat_line = extract_stats_from_df(os.path.join(dir_name, file))
        ep = int(get_prm('ep', file))
        res_df.append(pd.DataFrame(stat_line, index=[ep]))
        
    return pd.concat(res_df)

def plot_stats_ep(stats_df, hist_file, text='', text_coords=[0,0], limits={}):
    import pandas as pd
    import numpy as np
    import pickle
    from matplotlib import pyplot as plt
    
    stats_df = stats_df.sort_index()
    cats_colors = {'planck_z' : 'b', 'planck_no_z' : 'g', 'mcxcwp' : 'r', 'actwp' : 'c'}
    metr_colors = {'iou' : 'r', 'dice' : 'b', 'loss' : 'c'}

    hist = None
    with open(hist_file, 'rb') as f:
        d = pickle.load(f)
        hist = pd.DataFrame(d, index=np.arange(1, len(d['loss']) + 1))

    min_ep = max(hist.index.max(), stats_df.index.max())
    
    fig, ax = plt.subplots(4, 1, figsize=(12,18), sharex=True)
    plt.subplots_adjust(hspace=0.05)
    
    ####         recalls         ####
    for cat in cats_colors:
        line, = ax[0].plot(stats_df.index, stats_df[cat+'_recall'], cats_colors[cat]+'o-',
                          markersize=5)
        line.set_label(cat)
        
    ####         fp              ####
    line, = ax[1].plot(stats_df.index, stats_df['fp'], 'ko-', markersize=5)
    line.set_label('fp')
    
    ####        history          ####
    for metr in metr_colors:
        i = 2
        if metr == 'loss':
            i = 3
        line, = ax[i].plot(hist.index, hist[metr], metr_colors[metr]+'o-',
                          markersize=5)
        line.set_label(metr)
        line, = ax[i].plot(hist.index, hist['val_' + metr], metr_colors[metr]+'o-',
                          markersize=5, alpha=0.5)
        line.set_label('val_'+metr)
    
    #### ticks, grid, labels, legend ####
    ax[0].set_yticks(np.arange(0, 1.1, 0.1), minor=True)
    ax[0].set_yticks(np.arange(0, 1.1, 0.2))
    if not (fp in limits):
        limits['fp'] = [min(stats_df['fp']), max(stats_df['fp'])]
        fp_min = limits['fp'][0]
        limits['fp'][0] = (fp_min // 100) * 100
    ax[1].set_yticks(np.arange(*limits['fp'], 50), minor=True)
    ax[1].set_yticks(np.arange(*limits['fp'], 100))
    if iou_max is None:
        iou_max = max(hist['iou'])
    ax[2].set_yticks(np.arange(0, iou_max+0.01, 0.005), minor=True)
    ax[2].set_yticks(np.arange(0, iou_max+0.01, 0.01))
    if loss_max is None:
        loss_max = max(hist['loss'])
    ax[3].set_yticks(np.arange(0, loss_max, 0.05), minor=True)
    ax[3].set_yticks(np.arange(0, loss_max, 0.1))

    
    for i in range(4):
        ax[i].legend()
        ax[i].set_xticks(stats_df.index[4::5])
        ax[i].set_xticks(stats_df.index, minor=True)
        ax[i].grid(True, axis='both', which='major')
        ax[i].grid(True, axis='both', which='minor', alpha=0.2)
    
    ax[3].set_xlabel('epochs')
    ax[0].set_ylabel('recall')
    ax[1].set_ylabel('fp')
    ax[2].set_ylabel('metrics')
    ax[3].set_ylabel('metrics')
    ax[1].text(*text_coords, text, c='r', size=20)
    
 
def plot_history(hist_file, loss_tick=None, cut_eps=[0, -1]):
    import pickle
    import pandas as pd
    import numpy as np
    from matplotlib import pyplot as plt
    
    hist = None
    with open(hist_file, 'rb') as f:
        hist = pickle.load(f)
        hist = pd.DataFrame(hist, index=np.arange(1, len(hist['loss']) + 1))
    _, ax = plt.subplots(2, 1, figsize=(10,14))
    
    hist = hist.iloc[cut_eps[0]:cut_eps[1]]
    
    line, = ax[0].plot(hist.index, hist['loss'], 'co-')
    line.set_label('loss')
    line, = ax[0].plot(hist.index, hist['val_loss'], 'co-', alpha=0.5)
    line.set_label('val_loss')
    if not (loss_tick is None):
        ax[0].set_xtick(loss_tick[::5])
        ax[0].set_xtick(loss_tick, minor=True)
    
    for c, metr in zip('rb', ['iou', 'dice']):
        line, = ax[1].plot(hist.index, hist[metr], c+'o-')
        line.set_label(metr)
        line, = ax[1].plot(hist.index, hist['val_'+metr], c+'o-', alpha=0.5)
        line.set_label('val_'+metr)
    
    for i in range(2):
        ax[i].legend()
        ax[i].set_xticks(hist.index, minor=True)
        ax[i].set_xticks(hist.index[4::5])
        ax[i].grid(True, axis='both', which='major')
        ax[i].grid(True, axis='both', which='minor', alpha=0.2)
        

def mark_cat(cat, d):
    import numpy as np
    import pandas as pd
    from DS_healpix_fragmentation import radec2pix
    
    pix2 = radec2pix(cat['RA'], cat['DEC'], 2)
    cat['pix2'] = ''
    for name in d:
        cat['pix2'].loc[np.in1d(pix2, d[name])] = name
    return cat

def colors_iterator():
    c = ['r', 'b', 'g', 'c', 'm', 'y', 'k', 'tab:blue', 'tab:orange', 'tab:green',
            'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive']

    while True:
        yield c[0]
        c = c[1:]

def get_cat_name(filename):
    import os

    return os.path.splitext(os.path.basename(filename))[0]

def this_ax(ax, ylim=[0, 1.05], yticks=[0.1, 1.1, 0.1], xlim=[0.1, 1], xtitle='max_pred'):
    import numpy as np

    #l = ax.axvline(0.65, c='brown', linestyle='-.')
    #l.set_label('max_pred=0.65')
    ax.set_xlabel(xtitle)
    ax.set_xticks(np.arange(0.1, 1.1, 0.1), minor=True)
    ax.set_xlim(*xlim)
    
    
    ax.set_yticks(np.arange(*yticks), minor=False)
    ax.set_ylim(*ylim)
    
    
    #ax.set_yticks(np.arange(*yticks)[::2], minor=False)
    ax.grid(True, axis='both', which='major', linestyle=':')
    ax.grid(True, axis='both', which='minor', alpha=0.2, linestyle=':')
    ax.legend()


def inter_cats(df_base, df_inter, rad=400/3600):
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    import numpy as np
    import pandas as pd

    sc_base = SkyCoord(ra=np.array(df_base['RA'])*u.degree, 
            dec=np.array(df_base['DEC'])*u.degree, frame='icrs')
    sc_inter = SkyCoord(ra=np.array(df_inter['RA'])*u.degree, 
            dec=np.array(df_inter['DEC'])*u.degree, frame='icrs')

    _, d2d, _ = sc_base.match_to_catalog_sky(sc_inter)
    matched = d2d.degree < rad
    
    df_base = df_base[matched]
    df_base.index = np.arange(len(df_base))
    return df_base

def dif_cats(df_base, df_inter, rad=400/3600):
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    import numpy as np
    import pandas as pd

    sc_base = SkyCoord(ra=np.array(df_base['RA'])*u.degree, 
            dec=np.array(df_base['DEC'])*u.degree, frame='icrs')
    sc_inter = SkyCoord(ra=np.array(df_inter['RA'])*u.degree, 
            dec=np.array(df_inter['DEC'])*u.degree, frame='icrs')

    _, d2d, _ = sc_base.match_to_catalog_sky(sc_inter)
    matched = d2d.degree < rad
    
    df_base = df_base[np.logical_not(matched)]
    df_base.index = np.arange(len(df_base))
    return df_base

class Counts_cat:
    def __init__(self, bins, name, found_counts, error_counts, alls, found_coef, err_coef):
        self.bins = bins
        self.name = name
        self.found_counts = found_counts
        self.error_counts = error_counts
        self.alls = alls
        self.found_coef = found_coef
        self.err_coef = err_coef
        self.found = found_counts/found_coef
        self.error = error_counts/err_coef
        
    def error_for_all(self):
        import numpy as np
        return np.nan_to_num(self.error_counts/self.alls)
    
    def err_ratio(self):
        import numpy as np
        return np.nan_to_num(self.error / self.found)
    def precision(self):
        import numpy as np
        return np.nan_to_num(self.found_counts / self.alls)
    def corr_precision(self):
        return (1 - self.err_ratio()) * self.precision()
def found_all_error_calculated(det_cat, true_cat, small_rads=[0, 400], big_rads=[1000, 1500],
        saving=None, n_bins=20, func_cat=None,
                              with_tqdm=False, name=''):
    import numpy as np
    import pandas as pd
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    from tqdm.notebook import tqdm
    
    def func(angle1, angle2, det_cat, true_cat):
        colored = []
        tr = SkyCoord(ra=np.array(true_cat['RA'])*u.degree, 
                dec=np.array(true_cat['DEC'])*u.degree, frame='icrs')
        
        iterator = range(len(det_cat))
        if with_tqdm:
            iterator = tqdm(iterator)

        for i in iterator:
            det = SkyCoord(ra=det_cat.loc[i, 'RA']*u.degree, dec=det_cat.loc[i, 'DEC']*u.degree, frame='icrs')
            sep = tr.separation(det).degree

            count = np.count_nonzero(np.logical_and(angle1/3600 <= sep, sep < angle2/3600))
            colored.extend(count * [det_cat.loc[i, 'max_pred']])

        return colored
    
    def square(rads):
        return np.pi * (rads[1] ** 2 - rads[0] ** 2)
    
    if not (func_cat is None):
        det_cat = func_cat(det_cat)
        true_cat = func_cat(true_cat)
    
    found = func(*small_rads, det_cat, true_cat)
    error = func(*big_rads, det_cat, true_cat)
    
    small_sq = square(small_rads)
    big_sq = square(big_rads)
    
    bins = np.arange(0, 1 + 1/n_bins, 1/n_bins)
    
    found_counts, _ = np.histogram(found, bins)
    error_counts, _ = np.histogram(error, bins)
    all_counts, _ = np.histogram(det_cat['max_pred'], bins)
    
    cc = Counts_cat(bins=bins, name=name, found_counts=found_counts, error_counts=error_counts, alls=all_counts, found_coef=small_sq, 
                   err_coef=big_sq)
    return cc
