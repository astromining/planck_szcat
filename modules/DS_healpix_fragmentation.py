def one_pixel_fragmentation(o_nside, o_pix, depth):
    import healpy as hp
    import numpy as np
    
    def recursive_fill(matr):
        if matr.shape[0] == 1:
            return

        mid = matr.shape[0] // 2
        np.left_shift(matr, 1, out=matr)
        matr[mid:, :] += 1

        np.left_shift(matr, 1, out=matr)
        matr[:, mid:] += 1

        for i in [0, mid]:
            for j in [0, mid]:
                recursive_fill(matr[i:i+mid, j:j+mid])
                
    m_len = 2 ** depth
    f_matr = np.full((m_len, m_len), o_pix)
    
    recursive_fill(f_matr)
    return f_matr


def find_biggest_pixel(ra, dec, radius, root_nside=1, max_nside=32):
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    import healpy as hp
    import numpy as np

    nside = root_nside
    radius = np.radians(radius)
    sc = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
    theta = sc.galactic.l.degree
    phi = sc.galactic.b.degree
    vec = hp.ang2vec(theta=theta, phi=phi, lonlat=True)
    
    pixels = hp.query_disc(vec=vec, nside=nside, radius = radius, inclusive=False, 
                                nest=True)
    while len(pixels) <= 1:
        if nside == max_nside:
            break
        nside *= 2
        pixels = hp.query_disc(vec=vec, nside=nside, radius = radius, inclusive=False, 
                                    nest=True)
    if nside > 1:
        nside //= 2
    return nside, hp.vec2pix(nside, *vec, nest=True)

def matr2dict(matr):
    d = {}
    for i in range(matr.shape[0]):
        for j in range(matr.shape[1]):
            d[matr[i, j]] = (i, j)
    return d

def radec2pix(ra, dec, nside):
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    import healpy as hp
    import numpy as np

    sc = SkyCoord(ra=np.array(ra)*u.degree, dec=np.array(dec)*u.degree, frame='icrs')
    return hp.ang2pix(nside, sc.galactic.l.degree, sc.galactic.b.degree, 
                                  nest=True, lonlat=True)
def pix2radec(ipix, nside):
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    import healpy as hp
    import numpy as np

    theta, phi = hp.pix2ang(nside, ipix=np.array(ipix), nest=True, lonlat=True)

    sc = SkyCoord(l=theta*u.degree, b=phi*u.degree, frame='galactic')
    return sc.icrs.ra.degree, sc.icrs.dec.degree     

def pix2pix(ipix, nside, nside_fin):
    import healpy as hp

    vec = hp.pix2vec(ipix=ipix, nside=nside, nest=True)
    return hp.vec2pix(nside_fin, *vec, nest=True)

def draw_circles_h(ra, dec, data, nside, mdict, shape, coef=0.02):
    import numpy as np
    from skimage.draw import circle

    coef = shape[0] * coef / max(data)
    pic = np.zeros(shape, dtype=np.uint8)
    pix = radec2pix(ra, dec, nside)
    for i in range(len(pix)):
        if pix[i] in mdict:
            x, y = mdict[pix[i]]
            pic[circle(x, y, data[i] * coef, shape=shape)] = 1
    
    return pic

def make_x_n(x, y, size, shape, th=3):
    import numpy as np
    size = int(size)
    coords = []
    coords.extend([(xx, yy) 
                   for xx in range(max(x-th,0), min(x+th+1,shape[0]-1)) 
                   for yy in range(max(y-size,0), min(y+size+1,shape[1]-1))])
    coords.extend([(xx, yy) 
                   for xx in range(max(x-size,0), min(x+size+1,shape[0]-1)) 
                   for yy in range(max(y-th,0), min(y+th+1,shape[1]-1))])
    coords = np.array(coords)
    if len(shape) > 2:
        return coords[:,0], coords[:,1], np.zeros(len(coords))
    return coords[:,0], coords[:,1]

def draw_x_h_n(ra, dec, data, nside, mdict, shape, coef=0.02):
    import numpy as np
    from skimage.draw import circle

    coef = shape[0] * coef / max(data)
    pic = np.zeros(shape, dtype=np.uint8)
    pix = radec2pix(ra, dec, nside)
    for i in range(len(pix)):
        if pix[i] in mdict:
            x, y = mdict[pix[i]]
            x_pic = make_x_n(x, y, data[i] * coef, shape=shape)
            pic[x_pic] = 1
    
    return pic

def draw_dots_h(ra, dec, data, nside, mdict, shape):
    import numpy as np

    pic = np.zeros(shape)
    pix = radec2pix(ra, dec, nside)
    coords = [mdict[p] for p in pix if p in mdict]
    if len(shape) == 2:
        for i in range(len(data)):
            x, y = coords[i]
            pic[x, y] = data[i]
    else:
        for i in range(len(data)):
            x, y = coords[i]
            pic[x, y, 0] = data[i]

    
    return pic

def draw_proper_circle_s(ras, decs, radius, nside, mdict, shape, centers_in_patch=False):
    import numpy as np
    res = np.zeros(shape)
    for ra, dec in zip(ras, decs):
        res = np.logical_or(res, draw_proper_circle(ra, dec, radius, nside, mdict, shape, 
            False, centers_in_patch))
    return res

def draw_proper_circle(ra, dec, radius, nside, mdict, shape, coords_mode=True, 
        centers_in_patch=False):
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    import healpy as hp
    import numpy as np

    sc = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
    vec = hp.ang2vec(theta=sc.galactic.l.degree, phi=sc.galactic.b.degree, lonlat=True)
    pix = hp.query_disc(nside, vec, np.radians(radius), nest=True, inclusive=True)
    coords = [mdict[p] for p in pix if p in mdict]
    if coords_mode:
        return np.array(coords)
    
    pic = np.zeros(shape, dtype=np.uint8)
    if centers_in_patch:
        cl_pix = hp.vec2pix(nside, *vec, nest=True)
        if not (cl_pix in mdict):
            return pic
    if len(shape) == 2:
        for x, y in coords:
            pic[x, y] = 1
    else:
        for x, y in coords:
            pic[x, y, 0] = 1
    return pic 

def nearest_power(n):
    k = 0
    isPower = True
    while n > 0:
        if n % 2 != 0 and n > 1:
            isPower = False
        n //= 2
        k += 1
    if isPower:
        k -= 1
    return 2**k

def zoom_to_circle(coords, matr, add_power=True):
    xmin = coords[:, 0].min()
    xmax = coords[:, 0].max()
    ymin = coords[:, 1].min()
    ymax = coords[:, 1].max()
    if not add_power:
        return matr[xmin:xmax, ymin:ymax]
    xdif = xmax - xmin
    ydif = ymax - ymin
    
    map_size = nearest_power(max(xdif, ydif))
    xdif = map_size - xdif
    ydif = map_size - ydif
    xmin -= xdif // 2 + xdif % 2
    ymin -= ydif // 2 + ydif % 2
    xmax += xdif // 2
    ymax += ydif // 2
    
    return matr[xmin:xmax, ymin:ymax]

def radec2vec(ra, dec):
    import numpy as np
    import healpy as hp
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    
    sc = SkyCoord(ra=np.array(ra)*u.degree, dec=np.array(dec)*u.degree, frame='icrs')
    return hp.ang2vec(theta=sc.galactic.l.degree, phi=sc.galactic.b.degree, lonlat=True)

def cut_cat_by_pix(df, big_pix):
    import numpy as np
    pix = radec2pix(df['RA'], df['DEC'], 2)
    df = df[np.in1d(pix, big_pix)]
    df.index = np.arange(len(df))
    return df
