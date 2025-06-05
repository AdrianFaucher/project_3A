import pandas as pd
import numpy as np

def parse_galaxy_data(file_path: str, 
                      ra_range: tuple = None, 
                      dec_range: tuple = (-70,0), 
                      dist_range: tuple = None) -> pd.DataFrame:
    """Read TRGB galaxy data, parse RA/DEC, compute distance, and apply optional filtering.

    Args:
        file_path (str): Path to the data file.
        ra_range (tuple, optional): (min_hour, max_hour) to filter RA in decimal hours.
        dec_range (tuple, optional): (min_deg, max_deg) to filter DEC in decimal degrees.
        dist_range (tuple, optional): (min_mpc, max_mpc) to filter distance in Mpc.

    Returns:
        pd.DataFrame: Processed and optionally filtered dataframe.
    """
    def parse_ra(ra_str):
        h = int(ra_str[0:2])
        m = int(ra_str[2:4])
        s = float(ra_str[4:])
        return h, m, s

    def parse_dec(dec_str):
        sign = -1 if dec_str.startswith('-') else 1
        dec_str = dec_str.lstrip('+-')
        d = int(dec_str[0:2])
        m = int(dec_str[2:4])
        s = float(dec_str[4:])
        return sign , d, m, s

    def dm_to_mpc(dm):
        return 10 ** ((dm - 25) / 5)

    # Load data
    df = pd.read_csv(file_path, comment='%', skiprows=6, header=None)
    df.columns = ['pgc', 'v', 'e_v', 'RA2000', 'DEC2000', 'DMtrgb', 'eDMt']

    # Parse RA and DEC (sexagesimal)
    df[['RAh', 'RAm', 'RAs']] = df['RA2000'].apply(lambda x: pd.Series(parse_ra(str(x).zfill(7))))
    df[[D,'DEd', 'DEm', 'DEs']] = df['DEC2000'].apply(lambda x: pd.Series(parse_dec(str(x).zfill(8))))

    # Compute decimal RA and DEC
    df['RA_decimal'] = df['RAh'] + df['RAm'] / 60 + df['RAs'] / 3600
    df['DEC_decimal'] = np.sign(df['DEd']) * (abs(df['DEd']) + df['DEm'] / 60 + df['DEs'] / 3600)

    # Compute distance
    df['Dis'] = df['DMtrgb'].apply(dm_to_mpc)

    # Apply optional filters
    if ra_range:
        df = df[(df['RA_decimal'] >= ra_range[0]) & (df['RA_decimal'] <= ra_range[1])]
    if dec_range:
        df = df[(df['DEC_decimal'] >= dec_range[0]) & (df['DEC_decimal'] <= dec_range[1])]
    if dist_range:
        df = df[(df['Dis'] >= dist_range[0]) & (df['Dis'] <= dist_range[1])]

    return df

