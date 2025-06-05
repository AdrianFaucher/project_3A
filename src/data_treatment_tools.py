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
    e_distance_center = float(df.loc[df["Name"]==galaxy_center,"e_Dis"].iloc[0])
    e_r_galaxy = df['e_Dis']
    
    e_dis_center = (e_r_galaxy * (r_galaxy+ distance_center * np.absolute(cos_theta)) + e_distance_center * (distance_center + r_galaxy * np.absolute(cos_theta)))/dis_center
    
    df['e_dis_center_'+galaxy_center] = e_dis_center
    
    if grid_incertainty:
        def calculate_min_max_distance_error(row):
            cos_theta=row["cos_theta_"+galaxy_center]
            def calculate_distance(distances:np.ndarray):
                rg , rc = distances[0],distances[1]
                return np.sqrt(np.square(rg) + np.square(rc) - 2 * rc * rg * cos_theta)
            
            x0=[row['Dis'],distance_center] # point central
            e_x0=[row['e_Dis'],e_distance_center] #erreur autour du point
            min_f, max_f = min_max_grid(calculate_distance,x0,e_x0) # calcule du max et du min de major infall sur l'espace défini
            f_x0 = calculate_distance(x0)
            e_min= f_x0 - min_f  # on veut les valeurs d'erreurs
            e_max= max_f - f_x0
            return pd.Series([e_min,e_max])
    

        result = df.apply(calculate_min_max_distance_error, axis=1)
        df['e_min_dis_center_'+galaxy_center]  = result[0]
        df['e_max_dis_center_'+galaxy_center] =  result[1]
    


def add_pertubative_distances(df:pd.DataFrame,galaxy_center:str)->None:
    """Add a new row to the galaxy dataframe containing pertubative distances between all the galaxies and a given galaxy_center 

    Args:
        df (pd.DataFrame): DataFrame containing the data about the galaxy
        galaxy_center (str): Name of the galaxy that will be used as the center of the cluster
    """
    Rc= float(df.loc[df["Name"]==galaxy_center,"Dis"].iloc[0])
    Rg ,theta = df['Dis'],df["angular_distance_"+galaxy_center]
    gamma= (Rg/Rc) - 1
    alpha=np.sin(theta/2)
    # Calculer distance
    dis_center = alpha*(2+gamma)*Rc #normalement il y a une valeur absolue sur le sin(theta) mais ici tout les sinus sont positif)
    # Ajouter la colonne au DataFrame
    df['pertubative_dis_center_'+galaxy_center] = dis_center
    
    # add incertainty
    e_Rc = float(df.loc[df["Name"]==galaxy_center,"e_Dis"].iloc[0])
    e_Rg = df['e_Dis']
    
    e_gamma=(e_Rg*Rc+e_Rc*Rg)/(Rc**2)
    
    e_dis_center = alpha*(e_Rc*(2+gamma)+Rc*(e_gamma))
    
    df['e_pertubative_dis_center_'+galaxy_center] = e_dis_center
    df['gamma']=gamma
    df['e_gamma']=e_gamma



### Minor infall


def add_minor_infall_velocity(df:pd.DataFrame,galaxy_center:str,grid_incertainty:bool=False)->None:
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
            return pd.Series([0, 0])  # prevent zero division
        
        numerator = (velocity_center * distance_center + vg * rg) - cos_theta* (vg * distance_center + velocity_center * rg)
        v_rad = numerator / rgcenter
        
        
        e_r1_barre = (rg*e_rgcenter+e_rg*rgcenter)/(rgcenter**2)
        e_r2_barre = (distance_center*e_rgcenter+e_distance_center*rgcenter)/(rgcenter**2)

        
        e_v_rad = (e_velocity_center * (distance_center + rg              * cos_theta)/rgcenter +
                  e_vg              * (rg              + distance_center * cos_theta)/rgcenter + 
                  e_distance_center * (velocity_center + vg              * cos_theta)          +
                  e_rg              * (vg              + velocity_center * cos_theta)
        )
    

        if not grid_incertainty:
            return pd.Series([v_rad,e_v_rad])
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
            e_x0=[e_rg,e_distance_center,e_vg,e_velocity_center] #erreur autour du point
            min_f, max_f = min_max_grid(f_minor_infall_bis,x0,e_x0) # calcule du max et du min de major infall sur l'espace défini
            f_x0 = f_minor_infall_bis(x0) 
            e_min= f_x0 - min_f  # on veut les valeurs d'erreurs
            e_max= max_f - f_x0
            return pd.Series([v_rad,e_v_rad,e_min,e_max])
    # caculate minor_infall_velocities Ajouter la colonne au DataFrame

    result = df.apply(calculate_minor_infall, axis=1)
    df['minor_infall_velocity_' + galaxy_center] = result[0]
    df['e_minor_infall_velocity_' + galaxy_center] = result[1]
    if grid_incertainty:
        df['e_min_minor_infall_velocity_' + galaxy_center] = result[2]
        df['e_max_minor_infall_velocity_' + galaxy_center] = result[3]






def add_pertubative_minor_infall_velocity(df: pd.DataFrame, galaxy_center: str) -> None:
    """Add a new column containing the new velocity regarding the galaxy center using the minor infall model.

    Args:
        df (pd.DataFrame): DataFrame containing the information about galaxy
        galaxy_center (str): Name of the galaxy center
    """
    distance_center = float(df.loc[df["Name"] == galaxy_center, "Dis"].iloc[0])
    velocity_center = float(df.loc[df["Name"] == galaxy_center, "VLG"].iloc[0])
    
    e_distance_center = float(df.loc[df["Name"] == galaxy_center, "e_Dis"].iloc[0])
    e_velocity_center = float(df.loc[df["Name"] == galaxy_center, "e_VLG"].iloc[0])
    
    # Function to calculate minor velocities
    def calculate_minor_infall(row):
        rg = row['Dis']
        vg = row['VLG']
        cos_theta = row['cos_theta_' + galaxy_center]
        theta = row['angular_distance_' + galaxy_center]
        rgcenter = row['pertubative_dis_center_' + galaxy_center]
        
        e_rg = row['e_Dis']
        e_vg = row['e_VLG']
        e_rgcenter = row['e_pertubative_dis_center_' + galaxy_center]
        
        gamma= row['gamma']
        e_gamma= row['e_gamma']
        
        #if rgcenter == 0:
        #    return pd.Series([0, 0])
        #
        #numerator = (velocity_center * distance_center + vg * rg) - cos_theta * (vg * distance_center + velocity_center * rg)
        #
        #e_r1 = (rg * e_rgcenter + e_rg * rgcenter) / (rgcenter ** 2)
        #e_r2 = (distance_center * e_rgcenter + e_distance_center * rgcenter) / (rgcenter ** 2)
        #error_velocity = (
        #    e_vg * (rg + cos_theta * distance_center) +
        #    e_velocity_center * (distance_center + cos_theta * rg)
        #) / rgcenter + e_r1 * (vg + velocity_center * cos_theta) + e_r2 * (velocity_center + vg * cos_theta)
        #
        #return pd.Series([numerator / rgcenter, error_velocity])
        beta= 1/(2*np.absolute(np.sin(theta/2)))
        v_rad=beta*(1-cos_theta)*(vg+velocity_center)-0.5*gamma*(1+cos_theta)*(vg-velocity_center)
        e_v_rad= beta*((np.absolute(1-cos_theta)+np.absolute(0.5*gamma*(1+cos_theta)))*(e_vg + e_velocity_center) +
                       0.5*e_gamma*np.absolute((1+cos_theta)*(vg-velocity_center)))
        return pd.Series([v_rad,e_v_rad])
    
    result = df.apply(calculate_minor_infall, axis=1)
    df['pertubative_minor_infall_velocity_' + galaxy_center] = result[0]
    df['e_pertubative_minor_infall_velocity_' + galaxy_center] = result[1]
    


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
    
def add_major_infall_velocity_bis(df:pd.DataFrame,galaxy_center:str, grid_incertainty:bool=False)->None:
    """add a new row containing the new velocity regarding the galaxy center using the major infall model on the second galaxy

    Args:
        df (pd.DataFrame): DataFrame containing the information about galaxy
        galaxy_center (str): name of the galaxy center

    """
    distance_center= float(df.loc[df["Name"]==galaxy_center,"Dis"].iloc[0])
    velocity_center = float(df.loc[df["Name"]==galaxy_center,"VLG"].iloc[0])
    
    e_distance_center = float(df.loc[df["Name"] == galaxy_center, "e_Dis"].iloc[0])
    e_velocity_center = float(df.loc[df["Name"] == galaxy_center, "e_VLG"].iloc[0])
    
    ## fonction to calculate minor velocities
    def calculate_major_infall_incertainty_bis(row):
        """
        calculate minor infall and the incertainty and send back the according new row
        """
        rg, vg, cos_theta, rgcenter = row['Dis'], row['VLG'], row['cos_theta_'+galaxy_center], row['dis_center_'+galaxy_center]
        e_rg, e_vg, e_rgcenter = row['e_Dis'],row['e_VLG'],row['e_dis_center_' + galaxy_center]

        numerator = velocity_center - vg * cos_theta
        denominator =  distance_center - rg * cos_theta

        # prevent dividing by 0
        if denominator == 0 or rgcenter == 0 :
            return pd.Series([0, 0])
        
        v_rad = (numerator / denominator) * rgcenter

        r1=rg
        r2=distance_center
        e_r1_barre = (rg*e_rgcenter+e_rg*rgcenter)/(rgcenter**2)
        e_r2_barre = (distance_center*e_rgcenter+e_distance_center*rgcenter)/(rgcenter**2)

        e_v_rad = (((e_velocity_center + e_vg * cos_theta)/np.absolute(denominator)) * rgcenter  +
                   (np.absolute(numerator)*(e_r2_barre + e_r1_barre * cos_theta)/np.square(denominator)) * rgcenter**2   
        )
        
        if not grid_incertainty:
            return pd.Series([v_rad,e_v_rad])
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
                
            def major_infall_analytic_incertainty(point, errors, cos_theta_val):
                """Calculate the major infall velocity with analytical uncertainty using partial derivatives.

                Args:
                    point (np.ndarray): [r1, r2, v1, v2]
                    errors (np.ndarray): [e_r1, e_r2, e_v1, e_v2]
                    cos_theta_val (float): Value of cos_theta

                Returns:
                    tuple: (f_value, lower_error, upper_error)
                """
                r1_val, r2_val, v1_val, v2_val = point
                e_r1, e_r2, e_v1, e_v2 = errors

                # Calculate f_value first
                rgcenter_val = np.sqrt(r1_val**2 + r2_val**2 - 2 * cos_theta_val * r1_val * r2_val)
                numerator_val = v2_val - v1_val * cos_theta_val
                denominator_val = r2_val - r1_val * cos_theta_val

                if denominator_val == 0:
                    return 0, 0, 0

                f_val = (numerator_val / denominator_val) * rgcenter_val

                # Calculate partial derivatives at the given point
                # These are the exact analytical derivatives:

                # ∂f/∂r1
                df_dr1_val = (numerator_val/denominator_val) * (r1_val - cos_theta_val * r2_val) / rgcenter_val - \
                             (numerator_val * cos_theta_val) / denominator_val**2 * rgcenter_val

                # ∂f/∂r2
                df_dr2_val = (numerator_val/denominator_val) * (r2_val - cos_theta_val * r1_val) / rgcenter_val + \
                             (numerator_val) / denominator_val**2 * rgcenter_val

                # ∂f/∂v1
                df_dv1_val = -cos_theta_val * rgcenter_val / denominator_val

                # ∂f/∂v2
                df_dv2_val = rgcenter_val / denominator_val

                # Calculate uncertainty using error propagation formula:
                # σ²_f = (∂f/∂r1)² * σ²_r1 + (∂f/∂r2)² * σ²_r2 + (∂f/∂v1)² * σ²_v1 + (∂f/∂v2)² * σ²_v2
                variance = (df_dr1_val**2 * e_r1**2 + 
                            df_dr2_val**2 * e_r2**2 + 
                            df_dv1_val**2 * e_v1**2 + 
                            df_dv2_val**2 * e_v2**2)

                uncertainty = np.sqrt(variance)

                return f_val, variance,
                   
            x0=[rg,distance_center,vg,velocity_center] # point central
            e_x0=[e_rg,e_distance_center,e_vg,e_velocity_center] #erreur autour du point
            min_f, max_f = min_max_grid(f_major_infall_bis,x0,e_x0) # calcule du max et du min de major infall sur l'espace défini

            f_x0 = f_major_infall_bis(x0) 
            e_min= f_x0 - min_f  # on veut les valeurs d'erreurs
            e_max= max_f - f_x0
            
            
            _ ,e_analytic = major_infall_analytic_incertainty(x0,e_x0,cos_theta)
            
            
            return pd.Series([v_rad,e_v_rad,e_analytic,e_min,e_max])
        
        
    # caculate minor_infall_velocities Ajouter la colonne au DataFrame

    result = df.apply(calculate_major_infall_incertainty_bis, axis=1)
    df['major_infall_velocity_bis_' + galaxy_center] = result[0]
    df['e_major_infall_velocity_bis_' + galaxy_center] = result[1]
    if grid_incertainty:
        df['e_analytic_major_infall_velocity_bis_' + galaxy_center] = result[2]
        df['e_min_major_infall_velocity_bis_' + galaxy_center] = result[3]
        df['e_max_major_infall_velocity_bis_' + galaxy_center] = result[4]



def new_CoM_procedure(df,galaxy1,galaxy2,m1_barre,row_name:str=None,grid_incertainty:bool=False):
    if row_name is None:
        row_name = "CoM_"+galaxy1+"_"+galaxy2+"_"+str(m1_barre)
    add_CoM(df,galaxy1,galaxy2,m1_barre,row_name)
    add_angular_distance(df,galaxy_center=row_name)
    add_distances(df,galaxy_center=row_name,grid_incertainty=grid_incertainty)
    add_major_infall_velocity(df,galaxy_center=row_name)
    add_minor_infall_velocity(df,galaxy_center=row_name,grid_incertainty=grid_incertainty)
    add_major_infall_velocity_bis(df,galaxy_center=row_name,grid_incertainty=grid_incertainty)
    print(row_name)


