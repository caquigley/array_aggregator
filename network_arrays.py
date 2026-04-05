import numpy as np
from obspy.geodetics import gps2dist_azimuth
from obspy import UTCDateTime


def setup_coordinate_system(lat_list, lon_list):
    R = 6372.7976   # radius of the earth
    #lat_list = df['lat'].to_list()
    #lon_list = df['lon'].to_list()
    #lons  = np.array([tr.stats.coordinates.longitude for tr in st]) #silvio version
    #lats  = np.array([tr.stats.coordinates.latitude for tr in st]) #silvio version
    lats = np.array(lat_list) # cade version
    lons = np.array(lon_list) # cade version
    lon0  = lons.mean()*np.pi/180.0
    lat0  = lats.mean()*np.pi/180.0
    yx    = R*np.array([ lats*np.pi/180.0-lat0, (lons*np.pi/180.0-lon0)*np.cos(lat0) ]).T
    intsd = np.zeros([len(lons),len(lons)])
    ints_az= np.zeros([len(lons),len(lons)])
    for ii in range(len(lat_list[:-1])):
        for jj in range(ii+1,len(lat_list)):
            # intsd[i,j]=np.sqrt(np.square(yx[j][0]-yx[i][0])+np.square(yx[j][1]-yx[i][1]))
            tmp=gps2dist_azimuth(lats[ii],lons[ii],lats[jj],lons[jj])
            intsd[ii,jj]=tmp[0]
            ints_az[ii,jj]=tmp[1]

    return yx, intsd, ints_az


def inversion(times, lat_list, lon_list):
    import numpy as np
    


    ### CROSS CORRELATION SEGMENT-----------------------------
    num_sta = len(times)
    
    Cmax = []

    lags = []
    #times = df['time'].to_numpy()
    for i in range(num_sta - 1):
        for j in range(i + 1, num_sta):
            lags.append(UTCDateTime(times[i])-UTCDateTime(times[j])) #find lag times between events

    lags = np.array(lags)
    #Cmax = np.array(Cmax)
    #print('lags:', lags, 'length: ', len(lags))
    
    # Get geometry
    yx, intsd, ints_az = setup_coordinate_system(lat_list, lon_list)
    ds = intsd[np.triu_indices(num_sta, 1)]
    #print('ds:', ds, 'length', len(ds))
    az = ints_az[np.triu_indices(num_sta, 1)]
    #print('az:', len(az), az)
    Dm = (np.array([ds * np.cos(np.radians(az)), ds * np.sin(np.radians(az))]).T) 
    #print('Dm', len(Dm), Dm)
    dt = lags

    # Inversion
    Gmi = np.linalg.inv(Dm.T @ Dm)
    sv = Gmi @ Dm.T @ dt

    # Velocity & Azimuth
    velocity = 1 / np.sqrt(sv[0]**2 + sv[1]**2) 
    caz = velocity * sv[0]
    saz = velocity * sv[1]
    azimuth = np.degrees(np.arctan2(saz, caz)) % 360

    # Uncertainty using Szuberla and Olson (2004)
    H = Dm @ Gmi @ Dm.T
    I = np.eye(len(Dm))
    sig2dt = (dt @ (I - H) @ dt.T) / (len(Dm) - 2)

    cov_sv = sig2dt * Gmi
    sx, sy = sv[0], sv[1]
    v = velocity

    sig2vl = (sx**2 * cov_sv[0, 0] + sy**2 * cov_sv[1, 1] + 2 * sx * sy * cov_sv[0, 1]) * v**6
    sig2th = (sy**2 * cov_sv[0, 0] + sx**2 * cov_sv[1, 1] - 2 * sx * sy * cov_sv[0, 1]) * v**4

    rms = np.sqrt(np.mean((Dm @ sv - dt)**2))

    return velocity, azimuth, rms, Cmax, sig2th, sig2vl, sig2dt, dt