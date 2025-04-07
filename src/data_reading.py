import pandas as pd

def parse_galaxy_data(file_path:str)->pd.DataFrame:
    """read the data and convert theme to a df

    Args:
        file_path (str): path to the data

    Returns:
        pd.DataFrame:  dataframe containing the data
    """
    # Define the structure of the data
    columns = [
        ('Name', 1, 16),
        ('RAh', 18, 19),
        ('RAm', 21, 22),
        ('RAs', 24, 27),
        ('DE-', 29, 29),
        ('DEd', 30, 31),
        ('DEm', 33, 34),
        ('DEs', 36, 37),
        ('T', 39, 40),
        ('Theta', 42, 45),
        ('VLG', 47, 49),
        ('e_VLG', 51, 52),
        ('Dis', 54, 57),
        ('e_Dis', 59, 62),
        ('f_Dis', 64, 66),
        ('Ref', 68, 71),
        ('Note', 73, 73)
    ]

    # Read the file and parse the data
    with open(file_path, 'r') as file:
        lines = file.readlines()

    galaxies = []
    for line in lines:
        if line.strip() and not line.startswith('='):
            galaxy_data = {}
            for col_name, start, end in columns:
                value = line[start-1:end].strip()
                galaxy_data[col_name] = value
            galaxies.append(galaxy_data)

    # Convert to DataFrame
    df = pd.DataFrame(galaxies)
    df = df[df["VLG"] != '']
    df = df[df["Dis"] != '']
    return df
