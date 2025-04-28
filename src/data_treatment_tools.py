import pandas as pd
import numpy as np
from src.tools import equatorial_to_cartesian,cartesian_to_equatorial


def add_radian_columns(df:pd.DataFrame)->None:
    """convert the data to make theme usable :
        - right ascension converted to degree/radian (new column added)
        - declinaison converted to degree/radian (new column added)
        - Dis, e-Dis, VLG, e-VLg converted to float

    Args:
        df (pd.DataFrame): data where the data are stored
    """
    # Convertir RA et Dec en radians et ajouter en tant que nouvelles colonnes
    df['RAh'] = df['RAh'].astype(int)
    df['RAm'] = df['RAm'].astype(int)
    df['RAs'] = df['RAs'].astype(float)
    df['DEd'] = df['DEd'].astype(int)
    df['DEm'] = df['DEm'].astype(int)
    df['DEs'] = df['DEs'].astype(float)

    df['RA_degrees'] = 15 * (df['RAh'] + df['RAm'] / 60 + df['RAs'] / 3600)
    df['RA_radians'] = np.deg2rad(df['RA_degrees'])
    df['RA_radians'] = pd.to_numeric(df['RA_radians'], errors='coerce')


    df['Dec_degrees'] = df['DEd'] + df['DEm'] / 60 + df['DEs'] / 3600
    df['Dec_degrees'] = df.apply(lambda row: -row['Dec_degrees'] if row['DE-'] == '-' else row['Dec_degrees'], axis=1)
    df['Dec_radians'] = np.deg2rad(df['Dec_degrees'])

    df['Dis'] = pd.to_numeric(df['Dis'], errors='coerce')
    df['e_Dis'] = pd.to_numeric(df['e_Dis'], errors='coerce')
    df['VLG'] = pd.to_numeric(df['VLG'], errors='coerce')
    df['e_VLG'] = pd.to_numeric(df['e_VLG'], errors='coerce')
    




def add_CoM(df:pd.DataFrame,galaxy1:str,galaxy2:str,m1_barre:float,row_name:str)->None:
    """Add a ne row to the dataframe containing the data of the Center of Mass of two given galaxies with a spcify mass_ratio
    between those two galaxies. This creat a new row in the dataframe with the name CoM_galaxy1_galaxy2_massRatio1

    Args:
        df (pd.DataFrame): DataFrame containing the data
        galaxy1 (str): Name of the first galaxy
        galaxy2 (str): Name of the second galaxy
        m1_barre (float): mass ratio between the galaxy ( here m1_barre = m1/(m1+m2) so m2_barre = m2/(m1+m2) = 1-m1_barre)
    """
    m2_barre = 1- m1_barre
    d1, ra1, dec1, v1 = float(df.loc[df["Name"]==galaxy1,"Dis"].iloc[0]), float(df.loc[df["Name"]==galaxy1,"RA_radians"].iloc[0]), float(df.loc[df["Name"]==galaxy1,"Dec_radians"].iloc[0]), float(df.loc[df["Name"]==galaxy1,"VLG"].iloc[0])
    
    d2, ra2, dec2, v2 = float(df.loc[df["Name"]==galaxy2,"Dis"].iloc[0]), float(df.loc[df["Name"]==galaxy2,"RA_radians"].iloc[0]), float(df.loc[df["Name"]==galaxy2,"Dec_radians"].iloc[0]), float(df.loc[df["Name"]==galaxy2,"VLG"].iloc[0])
    
    e_d1, e_d2, e_v1, e_v2 =float(df.loc[df["Name"]==galaxy1,"e_Dis"].iloc[0]), float(df.loc[df["Name"]==galaxy2,"e_Dis"].iloc[0]) ,float(df.loc[df["Name"]==galaxy1,"e_VLG"].iloc[0]), float(df.loc[df["Name"]==galaxy2,"e_VLG"].iloc[0])
    
    
    r1 = equatorial_to_cartesian(d1,ra1,dec1)
    r2 = equatorial_to_cartesian(d2,ra2,dec2)
    rc = m1_barre*r1 + m2_barre*r2
    new_coord = cartesian_to_equatorial(rc[0],rc[1],rc[2])
    
    e_dist = m1_barre * e_d1 + (1-m1_barre) * e_d2
    
    V1 = equatorial_to_cartesian(v1,ra1,dec1)
    V2 = equatorial_to_cartesian(v2,ra2,dec2)
    Vc = m1_barre*V1 + m2_barre*V2
    
    
    # Attention:  the new velocity is not necesseraly radial
    new_Rad_velocity = np.dot(Vc,rc)/ np.linalg.norm(rc)
    
    e_new_Rad_velocity =  m1_barre * e_v1 + m2_barre * e_v2

    if not df['Name'].isin([row_name]).any():
        new_row = {'Name' :row_name, 'RAh':None, 'RAm':None, 'RAs':None, 'DE-':None, 'DEd':None, 'DEm':None, 'DEs':None, 'T':None, 'Theta':None, 'VLG':new_Rad_velocity, 'e_VLG':e_new_Rad_velocity, 'Dis':new_coord[0], 'e_Dis':e_dist, 'f_Dis':None, 'Ref':None, 'Note':None, 'RA_degrees':None, 'RA_radians':new_coord[1], 'Dec_degrees':None, 'Dec_radians': new_coord[2]}    
        df.loc[len(df)] = new_row
    else:
        df.loc[df['Name'] == row_name, 'Dis' ] = new_coord[0]
        df.loc[df['Name'] == row_name, 'VLG' ] = new_Rad_velocity
        df.loc[df['Name'] == row_name, 'RA_radians' ] = new_coord[1]
        df.loc[df['Name'] == row_name, 'Dec_radians' ] = new_coord[2]
      
      
      
      
      
        
def add_angular_distance(df:pd.DataFrame,galaxy_center:str)->None:
    """add a new row to the galaxy dataframe containing the angular distances between all the galaxies and a given galaxy_center 

    Args:
        df (pd.DataFrame): DataFrame containing the data about the galaxy
        galaxy_center (str): Name of the galaxy that will be used as the center of the cluster
    """
    ra_center= float(df.loc[df["Name"]==galaxy_center,"RA_radians"].iloc[0])
    dec_center = float(df.loc[df["Name"] == galaxy_center, "Dec_radians"].iloc[0])
    # Calculer les composantes trigonométriques
    sin_dec_1 = np.sin(dec_center)
    sin_dec_2 = np.sin(df['Dec_radians'])
    cos_dec_1 = np.cos(dec_center)
    cos_dec_2 = np.cos(df['Dec_radians'])
    cos_ra_diff = np.cos(ra_center - df['RA_radians'])
    # Calculer cos(θ)
    cos_theta = sin_dec_1 * sin_dec_2 + cos_dec_1 * cos_dec_2 * cos_ra_diff
    # Ajouter la colonne au DataFrame
    df["cos_theta_"+galaxy_center]= cos_theta
    df['angular_distance_'+galaxy_center] = np.arccos(cos_theta)

def add_distances(df:pd.DataFrame,galaxy_center:str)->None:
    """add a new row to the galaxy dataframe containing distances between all the galaxies and a given galaxy_center 

    Args:
        df (pd.DataFrame): DataFrame containing the data about the galaxy
        galaxy_center (str): Name of the galaxy that will be used as the center of the cluster
    """
    distance_center= float(df.loc[df["Name"]==galaxy_center,"Dis"].iloc[0])
    Rg ,cos_theta = df['Dis'],df["cos_theta_"+galaxy_center]
    # Calculer distance
    dis_center = np.sqrt(np.square(Rg) + np.square(distance_center) - 2 * distance_center * Rg * cos_theta)
    # Ajouter la colonne au DataFrame
    df['dis_center_'+galaxy_center] = dis_center
    
    # add incertainty
    e_distance_center = float(df.loc[df["Name"]==galaxy_center,"e_Dis"].iloc[0])
    e_Rg = df['e_Dis']
    
    e_dis_center = (e_Rg * (Rg+ distance_center * np.absolute(cos_theta)) + e_distance_center * (distance_center + Rg * np.absolute(cos_theta)))/dis_center
    
    df['e_dis_center_'+galaxy_center] = e_dis_center







def add_minor_infall_velocity(df:pd.DataFrame,galaxy_center:str)->None:
    """add a new row containing the new velocity regarding the galaxy center using the minor infall model

    Args:
        df (pd.DataFrame): DataFrame containing the information about galaxy
        galaxy_center (str): name of the galaxy center

    """
    distance_center= float(df.loc[df["Name"]==galaxy_center,"Dis"].iloc[0])
    velocity_center = float(df.loc[df["Name"]==galaxy_center,"VLG"].iloc[0])
    
    e_distance_center = float(df.loc[df["Name"] == galaxy_center, "e_Dis"].iloc[0])
    e_velocity_center = float(df.loc[df["Name"] == galaxy_center, "e_VLG"].iloc[0])
    
    ## fonction to calculate minor velocities
    def calculate_minor_infall(row):
        rg, vg, cos_theta, rgcenter = row['Dis'], row['VLG'], row['cos_theta_'+galaxy_center], row['dis_center_'+galaxy_center]
        e_rg, e_vg, e_rgcenter = row['e_Dis'],row['e_VLG'],row['e_dis_center_' + galaxy_center]

        
        if rgcenter == 0:
            return 0  # prevent zero division
        
        numerator = (velocity_center * distance_center + vg * rg) - cos_theta* (vg * distance_center + velocity_center * rg)
        v_rad=numerator / rgcenter
        
        
        e_r1_barre = (rg*e_rgcenter+e_rg*rgcenter)/(rgcenter**2)
        e_r2_barre = (distance_center*e_rgcenter+e_distance_center*rgcenter)/(rgcenter**2)

        
        e_v_rad = (e_velocity_center * (distance_center + rg              * cos_theta)/rgcenter +
                  e_vg              * (rg              + distance_center * cos_theta)/rgcenter + 
                  e_distance_center * (velocity_center + vg              * cos_theta)          +
                  e_rg              * (vg              + velocity_center * cos_theta)
        )
        
        return pd.Series([v_rad,e_v_rad])
    
    result = df.apply(calculate_minor_infall, axis=1)
    df['minor_infall_velocity_' + galaxy_center] = result[0]
    df['e_minor_infall_velocity_' + galaxy_center] = result[1]



def add_major_infall_velocity(df:pd.DataFrame,galaxy_center:str)->None:
    """add a new row containing the new velocity regarding the galaxy center using the major infall model on the center_galaxy

    Args:
        df (pd.DataFrame): DataFrame containing the information about galaxy
        galaxy_center (str): name of the galaxy center

    """
    distance_center= float(df.loc[df["Name"]==galaxy_center,"Dis"].iloc[0])
    velocity_center = float(df.loc[df["Name"]==galaxy_center,"VLG"].iloc[0])
    
    ## fonction to calculate minor velocities
    def calculate_major_infall(row):
        rg, vg, cos_theta, rgcenter = row['Dis'], row['VLG'], row['cos_theta_'+galaxy_center], row['dis_center_'+galaxy_center]
        numerator = vg - velocity_center * cos_theta
        denominator = rg - distance_center * cos_theta

        # prevent dividing by 0
        if denominator == 0:
            return 0

        return (numerator / denominator) * rgcenter

    # caculate minor_infall_velocities Ajouter la colonne au DataFrame
    df['major_infall_velocity_'+galaxy_center] = df.apply(calculate_major_infall, axis=1)
    
def add_major_infall_velocity_bis(df:pd.DataFrame,galaxy_center:str)->None:
    """add a new row containing the new velocity regarding the galaxy center using the major infall model on the second galaxy

    Args:
        df (pd.DataFrame): DataFrame containing the information about galaxy
        galaxy_center (str): name of the galaxy center

    """
    distance_center= float(df.loc[df["Name"]==galaxy_center,"Dis"].iloc[0])
    velocity_center = float(df.loc[df["Name"]==galaxy_center,"VLG"].iloc[0])
    
    ## fonction to calculate minor velocities
    def calculate_major_infall_bis(row):
        rg, vg, cos_theta, rgcenter = row['Dis'], row['VLG'], row['cos_theta_'+galaxy_center], row['dis_center_'+galaxy_center]
        numerator = velocity_center - vg * cos_theta
        denominator =  distance_center - rg * cos_theta

        # prevent dividing by 0
        if denominator == 0:
            return 0

        return (numerator / denominator) * rgcenter

    # caculate minor_infall_velocities Ajouter la colonne au DataFrame
    df['major_infall_velocity_bis_'+galaxy_center] = df.apply(calculate_major_infall_bis, axis=1)
    
    
def new_CoM_procedure(df,galaxy1,galaxy2,m1_barre,row_name:str=None):
    if row_name is None:
        row_name = "CoM_"+galaxy1+"_"+galaxy2+"_"+str(m1_barre)
    add_CoM(df,galaxy1,galaxy2,m1_barre,row_name)
    add_angular_distance(df,galaxy_center=row_name)
    add_distances(df,galaxy_center=row_name)
    add_major_infall_velocity(df,galaxy_center=row_name)
    add_minor_infall_velocity(df,galaxy_center=row_name)
    add_major_infall_velocity_bis(df,galaxy_center=row_name)
    print(row_name)
