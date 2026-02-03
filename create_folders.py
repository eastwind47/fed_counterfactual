import os
import shutil

def copy_all_to_subfolders(src_folder, dest_folder, subfolders):
    for sub in subfolders:
        subfolder_path = os.path.join(dest_folder, sub)
        os.makedirs(subfolder_path, exist_ok=True)
        # Copy everything from src_folder into subfolder_path
        for item in os.listdir(src_folder):
            s = os.path.join(src_folder, item)
            d = os.path.join(subfolder_path, item)
            if os.path.isdir(s):
                if os.path.exists(d):
                    shutil.rmtree(d)
                shutil.copytree(s, d)
            else:
                shutil.copy2(s, d)
        print(f"Copied all contents to {subfolder_path}")

# Example usage:
# src_folder = "/Users/home/Documents/naz/research_codes/uncert_prop/synthetic_exp/scalability/obs_dimension/dim_128/sigma_y_set1/sigma_y_10e_4"
# dest_folder = "/Users/home/Documents/naz/research_codes/uncert_prop/synthetic_exp/scalability/obs_dimension/dim_128/sigma_Amn_set1"
# subfolders = ['Amn_10e_6', 'Amn_10e_5', 'Amn_10e_4', 'Amn_10e_3', 'Amn_10e_2', 'Amn_10e_1', 'Amn_10e0', 'Amn_10e1']

src_folder = "/Users/home/Documents/naz/research_codes/uncert_prop/synthetic_exp/scalability/obs_dimension/dim_128/sigma_y_set1/sigma_y_10e_4"
dest_folder = "/Users/home/Documents/naz/research_codes/uncert_prop/synthetic_exp/scalability/obs_dimension/dim_128/sigma_theta_set1"
subfolders = ['theta_m_10e_6', 'theta_m_10e_5', 'theta_m_10e_4', 'theta_m_10e_3', 'theta_m_10e_2', 'theta_m_10e_1', 'theta_m_10e0', 'theta_m_10e1']
copy_all_to_subfolders(src_folder, dest_folder, subfolders)