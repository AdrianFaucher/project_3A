import pandas as pd
import numpy as np
from src.tools import equatorial_to_cartesian, cartesian_to_equatorial, min_max_grid


def add_radian_columns(df:pd.DataFrame)->None:
    """convert the data to make theme usable :
        - right ascension converted to degree/radian (new column added)
        - declinaison converted to degree/radian (new column added)
        - Dis, e-Dis, VLG, e-VLg converted to float

    Args:
        df (pd.DataFrame): data where the data are stored
    """
    # Convertir RA et Dec en radians et ajouter en tant que nouvelles colonnes
    df['RA'] = df['RA'].astype(float)
    df['Dec'] = df['Dec'].astype(float)

    df['RA_radians'] = df['RA'] * (np.pi/180)
    df['RA_radians'] = pd.to_numeric(df['RA_radians'], errors='coerce')


    df['Dec_radians'] = df['Dec'] * (np.pi/180)
    df['Dec_radians'] = np.deg2rad(df['Dec_radians'])

    df['Dis'] = pd.to_numeric(df['Dis'], errors='coerce')
    df['e_Dis_min'] = pd.to_numeric(df['e_Dis_min'], errors='coerce')
    df['e_Dis_max'] = pd.to_numeric(df['e_Dis_max'], errors='coerce')
    df['V_h'] = pd.to_numeric(df['V_h'], errors='coerce')
    df['e_V_h'] = pd.to_numeric(df['e_V_h'], errors='coerce')
    




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
    d1, ra1, dec1, v1 = float(df.loc[df["Name"]==galaxy1,"Dis"].iloc[0]), float(df.loc[df["Name"]==galaxy1,"RA_radians"].iloc[0]), float(df.loc[df["Name"]==galaxy1,"Dec_radians"].iloc[0]), float(df.loc[df["Name"]==galaxy1,"V_h"].iloc[0])
    
    d2, ra2, dec2, v2 = float(df.loc[df["Name"]==galaxy2,"Dis"].iloc[0]), float(df.loc[df["Name"]==galaxy2,"RA_radians"].iloc[0]), float(df.loc[df["Name"]==galaxy2,"Dec_radians"].iloc[0]), float(df.loc[df["Name"]==galaxy2,"V_h"].iloc[0])
    
    e_d1_min, e_d1_max  = float(df.loc[df["Name"]==galaxy1,"e_Dis_min"].iloc[0]), float(df.loc[df["Name"]==galaxy1,"e_Dis_max"].iloc[0])
    e_d2_min, e_d2_max  = float(df.loc[df["Name"]==galaxy2,"e_Dis_min"].iloc[0]), float(df.loc[df["Name"]==galaxy2,"e_Dis_max"].iloc[0])
    e_v1, e_v2          = float(df.loc[df["Name"]==galaxy1,"e_V_h"].iloc[0])    , float(df.loc[df["Name"]==galaxy2,"e_V_h"].iloc[0])
    
    r1 = equatorial_to_cartesian(d1,ra1,dec1)
    r2 = equatorial_to_cartesian(d2,ra2,dec2)
    rc = m1_barre*r1 + m2_barre*r2
    new_coord = cartesian_to_equatorial(rc[0],rc[1],rc[2])
    
    e_dist_min = m1_barre * e_d1_min + (1-m1_barre) * e_d2_min
    e_dist_max = m1_barre * e_d1_max + (1-m1_barre) * e_d2_max
    
    V1 = equatorial_to_cartesian(v1,ra1,dec1)
    V2 = equatorial_to_cartesian(v2,ra2,dec2)
    Vc = m1_barre*V1 + m2_barre*V2
    
    
    # Attention:  the new velocity is not necesseraly radial
    new_Rad_velocity = np.dot(Vc,rc)/ np.linalg.norm(rc)
    
    e_new_Rad_velocity =  m1_barre * e_v1 + m2_barre * e_v2
    if not df['Name'].isin([row_name]).any():
        new_row = {'Name' :row_name, 'PGC':None,  'RA':new_coord[1]*(180/np.pi),   'Dec':new_coord[2]*(180/np.pi), 'V_h':new_Rad_velocity, 'e_V_h':e_new_Rad_velocity,'ref_V_h':None, 'Dis':new_coord[0], 'e_Dis_min':e_dist_min,'e_Dis_max':e_dist_max,'ref_dis':None, 'RA_radians':new_coord[1], 'Dec_radians': new_coord[2]}    
        df.loc[len(df)] = new_row
    else:
        df.loc[df['Name'] == row_name, 'Dis' ] = new_coord[0]
        df.loc[df['Name'] == row_name, 'e_Dis_min' ] = e_dist_min
        df.loc[df['Name'] == row_name, 'e_Dis_max' ] = e_dist_max
        df.loc[df['Name'] == row_name, 'V_h' ] = new_Rad_velocity
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

def add_distances(df:pd.DataFrame,galaxy_center:str,grid_incertainty:bool=False)->None:
    """add a new row to the galaxy dataframe containing distances between all the galaxies and a given galaxy_center 

    Args:
        df (pd.DataFrame): DataFrame containing the data about the galaxy
        galaxy_center (str): Name of the galaxy that will be used as the center of the cluster
    """
    distance_center= float(df.loc[df["Name"]==galaxy_center,"Dis"].iloc[0])
    r_galaxy ,cos_theta = df['Dis'],df["cos_theta_"+galaxy_center]

    dis_center = np.sqrt(np.square(r_galaxy) + np.square(distance_center) - 2 * distance_center * r_galaxy * cos_theta)
    # Ajouter la colonne au DataFrame
    df['dis_center_'+galaxy_center] = dis_center
    
    # add incertainty
    e_distance_center_min = float(df.loc[df["Name"]==galaxy_center,"e_Dis_min"].iloc[0])
    e_distance_center_max = float(df.loc[df["Name"]==galaxy_center,"e_Dis_max"].iloc[0])
    e_r_galaxy_min = df['e_Dis_min']
    e_r_galaxy_max = df['e_Dis_max']
    
    e_dis_center_min = (e_r_galaxy_min * (r_galaxy+ distance_center * np.absolute(cos_theta)) + e_distance_center_min * (distance_center + r_galaxy * np.absolute(cos_theta)))/dis_center
    e_dis_center_max = (e_r_galaxy_max * (r_galaxy+ distance_center * np.absolute(cos_theta)) + e_distance_center_max * (distance_center + r_galaxy * np.absolute(cos_theta)))/dis_center
    
    df['e_dis_center_min_' + galaxy_center] = e_dis_center_min
    df['e_dis_center_max_' + galaxy_center] = e_dis_center_max
    
    if grid_incertainty:
        def calculate_min_max_distance_error(row):
            cos_theta=row["cos_theta_"+galaxy_center]
            def calculate_distance(distances:np.ndarray):
                rg , rc = distances[0],distances[1]
                return np.sqrt(np.square(rg) + np.square(rc) - 2 * rc * rg * cos_theta)
            
            x0=[row['Dis'],distance_center] # point central
            e_x0_min=[row['e_Dis_min'],e_distance_center_min] #erreur autour du point
            e_x0_max=[row['e_Dis_max'],e_distance_center_max] #erreur autour du point
            min_f, max_f = min_max_grid(calculate_distance,x0,(e_x0_min,e_x0_max)) # calcule du max et du min de la distance sur l'espace défini
            f_x0 = calculate_distance(x0)
            e_min= f_x0 - min_f  # on veut les valeurs d'erreurs
            e_max= max_f - f_x0
            return pd.Series([e_min,e_max])
    

        result = df.apply(calculate_min_max_distance_error, axis=1)
        df['e_dis_center_min_'+galaxy_center]  = result[0]
        df['e_dis_center_max_'+galaxy_center] =  result[1]
    



### Minor infall

def add_minor_infall_velocity(df:pd.DataFrame,galaxy_center:str,grid_incertainty:bool=False)->None:
    """add a new row containing the new velocity regarding the galaxy center using the minor infall model

    Args:
        df (pd.DataFrame): DataFrame containing the information about galaxy
        galaxy_center (str): name of the galaxy center

    """
    distance_center= float(df.loc[df["Name"]==galaxy_center,"Dis"].iloc[0])
    velocity_center = float(df.loc[df["Name"]==galaxy_center,"V_h"].iloc[0])
    
    e_distance_center_min = float(df.loc[df["Name"] == galaxy_center, "e_Dis_min"].iloc[0])
    e_distance_center_max = float(df.loc[df["Name"] == galaxy_center, "e_Dis_max"].iloc[0])
    
    e_velocity_center = float(df.loc[df["Name"] == galaxy_center, "e_V_h"].iloc[0])
    
    
    ## fonction to calculate minor velocities
    def calculate_minor_infall(row):
        rg, vg, cos_theta, rgcenter = row['Dis'], row['V_h'], row['cos_theta_'+galaxy_center], row['dis_center_'+galaxy_center]
        e_rg_min, e_rg_max, e_vg, e_rgcenter_min, e_rgcenter_max = row['e_Dis_min'], row['e_Dis_max'], row['e_V_h'], row['e_dis_center_min_' + galaxy_center], row['e_dis_center_max_' + galaxy_center]

        
        if rgcenter == 0:
            return pd.Series([0, 0])  # prevent zero division
        
        numerator = (velocity_center * distance_center + vg * rg) - cos_theta* (vg * distance_center + velocity_center * rg)
        v_rad = numerator / rgcenter
        
        


    

        if not grid_incertainty:
            e_r1_barre_min = (rg*e_rgcenter_min+e_rg_min*rgcenter)/(rgcenter**2)
            e_r1_barre_max = (rg*e_rgcenter_max+e_rg_max*rgcenter)/(rgcenter**2)
            e_r2_barre_min = (distance_center*e_rgcenter_min+e_distance_center_min*rgcenter)/(rgcenter**2)
            e_r2_barre_max = (distance_center*e_rgcenter_max+e_distance_center_max*rgcenter)/(rgcenter**2)
            
            e_v_rad_min = (e_velocity_center * (distance_center + rg              * cos_theta)/rgcenter +
                  e_vg                   * (rg              + distance_center * cos_theta)/rgcenter + 
                  e_distance_center_min  * (velocity_center + vg              * cos_theta)          +
                  e_rg_min               * (vg              + velocity_center * cos_theta)
            )
            e_v_rad_max = (e_velocity_center * (distance_center + rg              * cos_theta)/rgcenter +
                      e_vg                  * (rg               + distance_center * cos_theta)/rgcenter + 
                      e_distance_center_max * (velocity_center  + vg              * cos_theta)          +
                      e_rg_max              * (vg               + velocity_center * cos_theta)
            )
            return pd.Series([v_rad,e_v_rad_min,e_v_rad_max])
        else:
            def f_minor_infall_bis(point:np.ndarray)->float:
                """calculate the major infall velocity for parameters given in point
    
                Args:
                    point (np.ndarray): [r1,r2,v1,v2]
    
                Returns:
                    float: major infall velocity
                """
                r1, r2, v1, v2 = point[0],point[1],point[2],point[3]
                rgcenter = np.sqrt(r1**2 + r2**2 - 2 * cos_theta * r1 * r2)
                numerator = (v2 * r2 + v1 * r1) - cos_theta* (v1 * r2 + v2 * r1)
                

                # prevent dividing by 0
                if rgcenter == 0:
                    return 0
                else:
                    return numerator / rgcenter 
              
            x0=[rg,distance_center,vg,velocity_center] # point central
            e_x0_min=[e_rg_min,e_distance_center_min,e_vg,e_velocity_center]
            e_x0_max=[e_rg_max,e_distance_center_max,e_vg,e_velocity_center] #erreur autour du point
            min_f, max_f = min_max_grid(f_minor_infall_bis,x0,(e_x0_min,e_x0_max)) # calcule du max et du min de major infall sur l'espace défini
            f_x0 = f_minor_infall_bis(x0) 
            e_v_rad_min= f_x0 - min_f  # on veut les valeurs d'erreurs
            e_v_rad_max= max_f - f_x0
            return pd.Series([v_rad,e_v_rad_min,e_v_rad_max])
    # caculate minor_infall_velocities Ajouter la colonne au DataFrame

    result = df.apply(calculate_minor_infall, axis=1)
    df['minor_infall_velocity_' + galaxy_center] = result[0]
    df['e_minor_infall_velocity_min_' + galaxy_center] = result[1]
    df['e_minor_infall_velocity_max_' + galaxy_center] = result[2]




### Major infall
    
    
def add_major_infall_velocity(df:pd.DataFrame,galaxy_center:str, grid_incertainty:bool=False)->None:
    """add a new row containing the new velocity regarding the galaxy center using the major infall model on the second galaxy

    Args:
        df (pd.DataFrame): DataFrame containing the information about galaxy
        galaxy_center (str): name of the galaxy center

    """
    distance_center= float(df.loc[df["Name"]==galaxy_center,"Dis"].iloc[0])
    velocity_center = float(df.loc[df["Name"]==galaxy_center,"V_h"].iloc[0])
    
    e_distance_center_min = float(df.loc[df["Name"] == galaxy_center, "e_Dis_min"].iloc[0])
    e_distance_center_max = float(df.loc[df["Name"] == galaxy_center, "e_Dis_max"].iloc[0])
    e_velocity_center = float(df.loc[df["Name"] == galaxy_center, "e_V_h"].iloc[0])
    
    ## fonction to calculate minor velocities
    def calculate_major_infall_incertainty(row):
        """
        calculate minor infall and the incertainty and send back the according new row
        """
        rg, vg, cos_theta, rgcenter = row['Dis'], row['V_h'], row['cos_theta_'+galaxy_center], row['dis_center_'+galaxy_center]
        e_rg_min, e_rg_max, e_vg, e_rgcenter_min, e_rgcenter_max = row['e_Dis_min'],row['e_Dis_max'], row['e_V_h'], row['e_dis_center_min_' + galaxy_center], row['e_dis_center_max_' + galaxy_center]

        numerator = velocity_center - vg * cos_theta
        denominator =  distance_center - rg * cos_theta

        # prevent dividing by 0
        if denominator == 0 or rgcenter == 0 :
            return pd.Series([0, 0])
        
        v_rad = (numerator / denominator) * rgcenter

        r1=rg
        r2=distance_center

        
        if not grid_incertainty:
            e_r1_barre_min = (rg*e_rgcenter_min+e_rg_min*rgcenter)/(rgcenter**2)
            e_r1_barre_max = (rg*e_rgcenter_max+e_rg_max*rgcenter)/(rgcenter**2)
            e_r2_barre_min = (distance_center*e_rgcenter_min+e_distance_center_min*rgcenter)/(rgcenter**2)
            e_r2_barre_max = (distance_center*e_rgcenter_max+e_distance_center_max*rgcenter)/(rgcenter**2)

            e_v_rad_min = (((e_velocity_center + e_vg * np.absolute(cos_theta))/np.absolute(denominator)) * rgcenter  +
                       (np.absolute(numerator)*(e_r2_barre_min + e_r1_barre_min *  np.absolute(cos_theta))/np.square(denominator)) * rgcenter**2   
            )
            e_v_rad_max = (((e_velocity_center + e_vg *  np.absolute(cos_theta))/np.absolute(denominator)) * rgcenter  +
                       (np.absolute(numerator)*(e_r2_barre_max + e_r1_barre_max *  np.absolute(cos_theta))/np.square(denominator)) * rgcenter**2   
            )
        
            return pd.Series([v_rad,e_v_rad_min,e_v_rad_max])
        else:
            def f_major_infall_bis(point:np.ndarray)->float:
                """calculate the major infall velocity for parameters given in point
    
                Args:
                    point (np.ndarray): [r1,r2,v1,v2]
    
                Returns:
                    float: major infall velocity
                """
                r1, r2, v1, v2 = point[0],point[1],point[2],point[3]
                rgcenter    =  np.sqrt(r1**2 + r2**2 - 2 * cos_theta * r1 * r2)
                numerator   =  v2 - v1 * cos_theta
                denominator =  r2 - r1 * cos_theta

                # prevent dividing by 0
                if denominator == 0:
                    return 0
                else:
                    return (numerator / denominator) * rgcenter
                
                   
            x0=[rg,distance_center,vg,velocity_center] # point central
            e_x0_min=[e_rg_min,e_distance_center_min,e_vg,e_velocity_center] #erreur autour du point
            e_x0_max=[e_rg_max,e_distance_center_max,e_vg,e_velocity_center]
            min_f, max_f = min_max_grid(f_major_infall_bis,x0,(e_x0_min,e_x0_max)) # calcule du max et du min de major infall sur l'espace défini

            f_x0 = f_major_infall_bis(x0) 
            e_v_rad_min = f_x0 - min_f  # on veut les valeurs d'erreurs
            e_v_rad_max = max_f - f_x0
            
            
            
            return pd.Series([v_rad,e_v_rad_min,e_v_rad_max])
        
        
    # caculate minor_infall_velocities Ajouter la colonne au DataFrame

    result = df.apply(calculate_major_infall_incertainty, axis=1)
    df['major_infall_velocity_' + galaxy_center] = result[0]
    df['e_major_infall_velocity_min_' + galaxy_center] = result[1]
    df['e_major_infall_velocity_max_' + galaxy_center] = result[2]




def new_CoM_procedure(df,galaxy1,galaxy2,m1_barre,row_name:str=None,grid_incertainty:bool=False):
    if row_name is None:
        row_name = "CoM_"+galaxy1+"_"+galaxy2+"_"+str(m1_barre)
    add_CoM(df,galaxy1,galaxy2,m1_barre,row_name)
    add_angular_distance(df,galaxy_center=row_name)
    add_distances(df,galaxy_center=row_name,grid_incertainty=grid_incertainty)
    add_major_infall_velocity(df,galaxy_center=row_name)
    add_minor_infall_velocity(df,galaxy_center=row_name,grid_incertainty=grid_incertainty)
    print(row_name)


