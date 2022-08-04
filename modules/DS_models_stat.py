b20_coef = 0.06324110671936758

def stat_split_cats(files, big_pix=list(range(48))):
    import pandas as pd
    import numpy as np
    from DS_healpix_fragmentation import radec2pix

    res_df = []
    comp_df = []
    for model in files:
        df = pd.read_csv(files[model])
        df = df.iloc[np.in1d(radec2pix(df['RA'], df['DEC'], 2), big_pix)]
        df.index = np.arange(len(df))
        line = {}
        line_c = {}
        for cat in ['planck_z', 'planck_no_z', 'mcxcwp', 'actwp']:
            cur_df = df[df['catalog'] == cat]
            cur_df.index = np.arange(len(cur_df))
            line_c[cat] = np.count_nonzero(cur_df['status'] == 'tp')
            line[cat] = line_c[cat] / len(cur_df)
        line['fp'] = np.count_nonzero(df['status'] == 'fp')
        res_df.append(pd.DataFrame(line, index=[model]))
        comp_df.append(pd.DataFrame(line_c, index=[model]))
    res_df = pd.concat(res_df)
    comp_df = pd.concat(comp_df)
    return res_df, comp_df

def cut_cat(df, dict_cut = {}, 
           big_pix=None):
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    import numpy as np
    from DS_healpix_fragmentation import radec2pix, cut_cat_by_pix
 
    sc = SkyCoord(ra=np.array(df['RA'])*u.degree, 
                  dec=np.array(df['DEC'])*u.degree, frame='icrs')
    df['b'] = sc.galactic.b.degree
    df['l'] = sc.galactic.l.degree
    for prm in dict_cut:
        if prm == 'b':
            df = df[np.abs(df[prm]) >= dict_cut[prm][0]]
            df = df[np.abs(df[prm]) < dict_cut[prm][1]]
            df.index = np.arange(len(df))
        elif prm in list(df):
            df = df[df[prm] >= dict_cut[prm][0]]
            df = df[df[prm] < dict_cut[prm][1]]
            df.index = np.arange(len(df))
    
    if not (big_pix is None):
        df = cut_cat_by_pix(df, big_pix)
    
    return df

def stat_orig_cats(det_cats_dict, true_cats_dir, 
        dict_cut = {}, big_pix = None, match_dist=5/60, n_try=200, max_pred_lim=None, 
        recall_only=False, format_s=lambda x:x, no_err=False, 
        other_cats={'eROSITA' :'~/Data/SRGz/clusters/clusters1_east_val_edit.csv', 
            'PSZ2(z)' : '~/Data/clusters/planck_z.csv'}):
    import os
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    from DS_data_transformation import calc_error
    import numpy as np
    import pandas as pd
    
    true_cats_files = next(os.walk(true_cats_dir))[-1]
    true_cats_files = [os.path.join(true_cats_dir, file) for file in true_cats_files]
    
    true_cats = {os.path.splitext(os.path.basename(file))[0] : pd.read_csv(file) for file in true_cats_files}
    true_cats.update({name : pd.read_csv(other_cats[name]) for name in other_cats})
    det_cats = {name : 
                pd.read_csv(det_cats_dict[name]) for name in det_cats_dict}
    
    comp_df = []
    recall_df = []
    
    for name in det_cats:
        df = det_cats[name]
        if 'status' in list(df):
            df = df[df['status'] != 'fn']
        if not (max_pred_lim is None):
            df = df[df['max_pred'] >= max_pred_lim]
        df['found'] = False
        df.index = np.arange(len(df))
        if 'b' in dict_cut:
            det_cats[name] = cut_cat(df, {'b' : dict_cut['b']}, big_pix)
        else:
            det_cats[name] = cut_cat(df, {}, big_pix)
    for name in true_cats:
        true_cats[name] = cut_cat(true_cats[name], dict_cut, big_pix)
    
    for det_name in det_cats:
        det = det_cats[det_name]
        line = {}
        line_r = {}

        det_sc = SkyCoord(ra=np.array(det['RA'])*u.degree, 
                      dec=np.array(det['DEC'])*u.degree, frame='icrs') 

        for tr_name in true_cats: 
            tr = true_cats[tr_name]
            tr_sc = SkyCoord(ra=np.array(tr['RA'])*u.degree, 
                          dec=np.array(tr['DEC'])*u.degree, frame='icrs')
            
            idx, d2d, _ = tr_sc.match_to_catalog_sky(det_sc)
            matched = d2d.degree <= match_dist
            det['found'].iloc[idx[matched]] = True
            
            line[tr_name] = np.count_nonzero(matched)
            if not recall_only and not no_err:
                line[tr_name+'_err'], line[tr_name+'_std'] = calc_error(det, tr, n_try=n_try)

            line_r[tr_name] = format_s(line[tr_name] / len(tr))
            
        line['all'] = len(det)
        line['fp'] = np.count_nonzero(det['found'] == False)
        line_r['fp'] = line['fp']
        line_r['all'] = line['all']
        comp_df.append(pd.DataFrame(line, index=[det_name]))
        recall_df.append(pd.DataFrame(line_r, index=[det_name]))
        line = {}
    
    for tr_name in true_cats: 
        line[tr_name] = len(true_cats[tr_name])
        line[tr_name+'_err'] = 0
    line['fp'] = 0
    line['all'] = 0
    comp_df.append(pd.DataFrame(line, index=['all']))
    
    comp_df = pd.concat(comp_df)
    recall_df = pd.concat(recall_df)
    if recall_only:
        return recall_df
    return comp_df, recall_df

def make_histogram(ax, counts_list, bins, label_list=None, coef_list=None, log=True, 
        add_legend=True, title='', histtype='step', alpha=1, legend_loc="upper left", 
        linestyles=None, colors=None):
    from DS_data_transformation import colors_iterator
    if coef_list is None:
        coef_list = [1] * len(counts_list)
    if label_list is None:
        label_list = [''] * len(counts_list)
    if linestyles is None:
        linestyles = ["-"] * len(counts_list)
    
    ci = colors_iterator()
    for i, counts in enumerate(counts_list):
        c = next(ci)
        if colors is not None:
            c = colors[i]
        ax.hist(bins[:-1], bins, weights=counts * coef_list[i], 
                  histtype=histtype, log=log, label=label_list[i], color=c, alpha=alpha,
                  linestyle=linestyles[i])
    ax.set_title(title)
    if add_legend:
        ax.legend(loc=legend_loc)

def calc_corr_b(det_cat, true_cat, small_rads=[0, 400/3600], big_rads=[1000/3600, 1500/3600], with_tqdm=False, 
        get_coefs=False):
    import numpy as np
    import pandas as pd
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    from tqdm.notebook import tqdm
    
    def func(angle1, angle2, det_cat, true_cat, fast=False):

        count = 0
        tr = SkyCoord(ra=np.array(true_cat['RA'])*u.degree, 
                dec=np.array(true_cat['DEC'])*u.degree, frame='icrs')
        if fast:
            det = SkyCoord(ra=np.array(det_cat['RA'])*u.degree, 
                    dec=np.array(det_cat['DEC'])*u.degree, frame='icrs')
            _, d2d, _ = det.match_to_catalog_sky(tr)
            return np.count_nonzero(np.logical_and(angle1 <= d2d.degree, 
                d2d.degree < angle2))

        iterator = range(len(det_cat))
        if with_tqdm:
            iterator = tqdm(iterator)

        for i in iterator:
            det = SkyCoord(ra=det_cat.loc[i, 'RA']*u.degree, dec=det_cat.loc[i, 'DEC']*u.degree, frame='icrs')
            sep = tr.separation(det).degree

            count += np.count_nonzero(np.logical_and(angle1 <= sep, sep < angle2))

        return count 
    
    def square(rads):
        return np.pi * (rads[1] ** 2 - rads[0] ** 2)
    
    found = func(*small_rads, det_cat, true_cat)
    error = func(*big_rads, det_cat, true_cat)
    not_found = func(small_rads[1], big_rads[0], det_cat, true_cat, fast=True) / len(true_cat)
    
    found_sq = square(small_rads)
    err_sq = square(big_rads)

    error_coef = (error / err_sq) * (found_sq / found) 

    ans = (1 - error_coef) * (1 + not_found)

    if get_coefs:
        return ans, error_coef, not_found 
    
    return ans 

def do_all_stats(det_cat, true_cats, true_cats_sc=None, match_dist=5/60, spec_precision=[], 
        use_err=False):
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    import numpy as np
    import pandas as pd
    
    if true_cats_sc is None:
        true_cats_sc = {}
        for true_name in true_cats:
            df = true_cats[true_name]
            true_cats_sc[true_name] = SkyCoord(ra=np.array(df['RA'])*u.degree, 
                                               dec=np.array(df['DEC'])*u.degree, frame='icrs')
   
    if 'status' in list(det_cat):
        det_cat = det_cat[det_cat['status'] != 'fn']
        det_cat.index = np.arange(len(det_cat))
    det_cat['found'] = False
    for cat in spec_precision:
        det_cat['found_' + cat] = False
    
    det_cat_sc = SkyCoord(ra=np.array(det_cat['RA'])*u.degree, 
                         dec=np.array(det_cat['DEC'])*u.degree, frame='icrs')
    
    line_r = {}
    for true_name in true_cats: 
        if len(true_cats[true_name]) == 0:
            continue
        tr_sc = true_cats_sc[true_name]
        idx, d2d, _ = tr_sc.match_to_catalog_sky(det_cat_sc)
        matched = d2d.degree <= match_dist
        det_cat.loc[idx[matched], 'found'] = True
        n_matched = np.count_nonzero(matched)

        if use_err:
            corr_coef = calc_corr_b(det_cat, true_cats[true_name], small_rads=[0, match_dist])
            n_matched *= corr_coef 

        line_r[true_name] =  n_matched / len(true_cats[true_name])


        if true_name in spec_precision:
            det_cat.loc[idx[matched], 'found_' + true_name] = True
            n_true_matched = np.count_nonzero(det_cat['found_' + true_name])
            line_r['precision_' + true_name] = n_true_matched / len(det_cat)
            line_r['found_' + true_name] = n_true_matched
    
    line_r['found'] = np.count_nonzero(det_cat['found'])
    #line_r['recall']
    line_r['precision'] = line_r['found'] / len(det_cat)
    line_r['all'] = len(det_cat)
    return line_r

def change_df(df, format_s=lambda x:'{:.2f}'.format(x)):
    for prm in list(df):
        for i in df.index:
            df.loc[i, prm] = format_s(df.loc[i, prm])
    return df

def stat_cat(det_cats, orig=True, big_pix=list(range(48)), match_dist=5/60, dict_cut={}, 
        other_cats={'eROSITA' :'~/Data/SRGz/clusters/clusters1_b20_edit.csv', 
            'PSZ2(z)' : '~/Data/clusters/planck_z.csv',
            'all_true' : '~/Data/original_catalogs/csv/other/PSZ2(z)_MCXC_ACT_united.csv',
            'PSZ2_inter_all_true' : '~/Data/original_catalogs/csv/other/psz2_inter_all_true.csv',
            'PSZ2_inter_eROSITA' : '~/Data/original_catalogs/csv/other/psz2_inter_eROSITA.csv'},
        spec_precision=[], read_det_files=True, use_err=False):
    import os
    import numpy as np
    import pandas as pd
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    from DS_data_transformation import get_cat_name
    from DS_healpix_fragmentation import cut_cat_by_pix
    
    true_cats_dir = '/home/rt2122/Data/original_catalogs/csv/'
    if not orig:
        true_cats_dir = '/home/rt2122/Data/clusters/'
    
    files = next(os.walk(true_cats_dir))[-1]
    true_cats = {get_cat_name(file) : pd.read_csv(os.path.join(true_cats_dir, file)) 
            for file in files}
    true_cats.update({name : pd.read_csv(other_cats[name]) for name in other_cats})
    true_cats = {name : cut_cat(true_cats[name], dict_cut=dict_cut, big_pix=big_pix) 
            for name in true_cats}
    true_cats_sc = {name : SkyCoord(ra=np.array(true_cats[name]['RA'])*u.degree,
                                   dec=np.array(true_cats[name]['DEC'])*u.degree, frame='icrs') 
                                   for name in true_cats}
    
    recall_df = []
    
    for det_name in det_cats:
        det = det_cats[det_name]
        if read_det_files:
            det = pd.read_csv(det)
        det = cut_cat(det, dict_cut=dict_cut, big_pix=big_pix)
        if len(det) == 0:
            continue
        line = do_all_stats(det, true_cats, true_cats_sc, match_dist=match_dist,
                spec_precision=spec_precision, use_err=use_err)
        recall_df.append(pd.DataFrame(line, index=[det_name]))
    recall_df = pd.concat(recall_df)
    return recall_df

def simple_recall(det_cat, true_cat, match=400/3600):
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    import numpy as np
    
    det_sc = SkyCoord(ra=np.array(det_cat['RA'])*u.degree, 
            dec=np.array(det_cat['DEC'])*u.degree, frame='icrs')
    tr_sc = SkyCoord(ra=np.array(true_cat['RA'])*u.degree, 
            dec=np.array(true_cat['DEC'])*u.degree, frame='icrs')
    
    _, d2d, _ = tr_sc.match_to_catalog_sky(det_sc)
    return np.count_nonzero(d2d.degree < match) / len(true_cat)
