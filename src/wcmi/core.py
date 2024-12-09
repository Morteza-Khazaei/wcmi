import os
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.optimize import differential_evolution, least_squares, curve_fit
from sklearn.metrics import r2_score


from .oh import *
from .wcm import *
from .core import *





class VegParamCal:

    def __init__(self, S1_freq_GHz=5.405, S1_local_overpass_time=18, year=2020, dir_radar_sigma=None, dir_risma=None, aafc_croptype=[158,], 
                 risma_station=['MB5',], sensor_depth=[0, 5,], ssm_inv_thr=0.05, ssr_lt_36deg=0.8, ssr_gt_36deg=0.9):
        
        self.wcm_fname = f'wcm_param_{str(year)}_{self.list_to_str(aafc_croptype)}_{self.list_to_str(risma_station)}_{self.list_to_str(sensor_depth)}'
        
        self.k = self.wavenumber(S1_freq_GHz)
        
        if not dir_radar_sigma:
            backscatter_dir = 'datasets/backscatter'
        else:
            backscatter_dir = dir_radar_sigma
        
        if not dir_risma:
            risma_dir = 'datasets/RISMA'
        else:
            risma_dir = dir_risma
        
        backscatter_files = self.search_file(backscatter_dir, year_filter=year)

        # filter risma csv files based on risma_station
        self.wcm_param_ct_st_dp = {}
        for ct in aafc_croptype:
            wcm_param_ct = {}
            for rst in risma_station:
                risma_files = self.search_file(risma_dir, year_filter=year, station=rst)
                wcm_param_dp = {}
                default_wcm_params = None
                for dp in sensor_depth:
                    print(f'Croptype: {ct}, Station: {rst}, Depth: {dp}')
                    # read risma bulk csv
                    df_doy_depth = self.read_risma_bulk_csv(risma_files[0], S1_lot=S1_local_overpass_time, depth=dp)
                    # read radar backscatter csv
                    S1_sigma_df_ct = self.read_radar_backscatter(backscatter_files[0], croptype=ct)

                    if not default_wcm_params:
                        print(f'Calculate default wcm params for croptype: {ct}, station: {rst}, depth: {dp}')
                        default_wcm_params = self.cal_wcm_veg_param(
                            df_sigma=S1_sigma_df_ct, df_risma=df_doy_depth, ssm_inv_thr=ssm_inv_thr, default_wcm_params=default_wcm_params, 
                            ssr_lt_36deg=ssr_lt_36deg, ssr_gt_36deg=ssr_gt_36deg)
                    
                    wcm_param_dp[dp] = self.cal_wcm_veg_param(
                        df_sigma=S1_sigma_df_ct, df_risma=df_doy_depth, ssm_inv_thr=ssm_inv_thr, default_wcm_params=default_wcm_params, 
                        ssr_lt_36deg=ssr_lt_36deg, ssr_gt_36deg=ssr_gt_36deg)
                    
                    # set default_wcm_params none
                    default_wcm_params = None
                
                wcm_param_ct[rst] = wcm_param_dp
            self.wcm_param_ct_st_dp[ct] = wcm_param_ct

        return None

    def run(self, save_as_csv=False):

        # df = self.dict_to_df(self.wcm_param_ct_st_dp)
        
        if save_as_csv:
            df.to_csv(f'{self.wcm_fname}.csv', index=False)
            print(f'Saved WCM parameters df to {self.wcm_fname}.csv')

        return self.wcm_param_ct_st_dp

    def plot_wcm_param(self, df, param_to_plot=['A', 'B', 'C', 'D'], hue='angle (deg)', style='RISMA', palette='viridis', markers=True, dashes=False, save_graph=False):
        # Create a four-row, one-column grid of subplots
        fig, axes = plt.subplots(len(param_to_plot), 1, figsize=(10, 12), sharex=True)

        # Loop through each constant and create a lineplot
        for i, constant in enumerate(param_to_plot):
            sns.lineplot(ax=axes[i], x='doy', y=constant, hue=hue, style=style, palette=palette, data=df, markers=markers, dashes=dashes)
            axes[i].set_ylabel(constant, fontsize=14)

            if i == 1:
                axes[i].legend(loc='upper right', fontsize=10, ncols=3)
            else:
                axes[i].legend().remove()

        # Set the x-axis label for the bottom subplot
        axes[-1].set_xlabel('DOY')

        # Adjust the spacing between subplots
        plt.tight_layout()

        if save_graph:
            fig.savefig(f'{self.wcm_fname}.png', dpi=800, bbox_inches='tight')

        # Display the plot
        plt.show()

        return None

    def list_to_str(self, lst_obj):
        # check if list has one object
        if len(lst_obj) == 1:
            return str(lst_obj[0])
        else:
            return '_'.join(map(str, lst_obj))

    def wavenumber(self, freq):
        freq *= 1e9  # convert to Hz
        v = 299792458 * 1e2  # speed of light (cm/s)
        return (2 * np.pi * freq) / v

    def dict_to_df(self, wcm_param):
        
        # Create a DataFrame from the angle_doy dictionary
        df = pd.DataFrame(wcm_param).T
        
        # reset index inplace and rename it as CropType
        df.index.name = 'CropType'
        df.reset_index(inplace=True)

        # melt df based on RISMA
        df_melted = df.melt(id_vars=['CropType'], var_name='RISMA', value_name='constants')

        # Expand df_melted base on constants column
        df_melted = df_melted.join(df_melted['constants'].apply(pd.Series))

        # drop constants
        df_melted.drop(columns=['constants'], inplace=True)

        # melt df based on depth (cm)
        df_melted = df_melted.melt(id_vars=['CropType', 'RISMA'], var_name='depth (cm)', value_name='constants')

        # Expand df_melted base on constants column
        df_melted = df_melted.join(df_melted['constants'].apply(pd.Series))

        # drop constants
        df_melted.drop(columns=['constants'], inplace=True)

        # melt df based on doy
        df_melted = df_melted.melt(id_vars=['CropType', 'RISMA', 'depth (cm)'], var_name='doy', value_name='constants')

        # Expand df_melted base on constants column
        df_melted = df_melted.join(df_melted['constants'].apply(pd.Series))

        # drop constants
        df_melted.drop(columns=['constants'], inplace=True)
        # df_melted.head()

        # melt df based on angle
        df_melted = df_melted.melt(id_vars=['CropType', 'RISMA', 'depth (cm)', 'doy'], var_name='angle (deg)', value_name='constants')

        # Expand df_melted base on constants column
        df_melted = df_melted.join(df_melted['constants'].apply(pd.Series))

        df_melted['A'] = df_melted[0].apply(lambda x: x[0][0] if isinstance(x, list) else None)
        df_melted['B'] = df_melted[0].apply(lambda x: x[0][1] if isinstance(x, list) else None)
        df_melted['C'] = df_melted[0].apply(lambda x: x[1][0] if isinstance(x, list) else None)
        df_melted['D'] = df_melted[0].apply(lambda x: x[1][1] if isinstance(x, list) else None)
        df_melted['ssm'] = df_melted[1].apply(lambda x: x[0] if isinstance(x, list) else None)
        df_melted['ssr'] = df_melted[1].apply(lambda x: x[1] if isinstance(x, list) else None)

        df_melted.drop(columns=['constants', 0, 1], inplace=True)

        return df_melted
    
    def mergeDictionary(self, dict1, dict2):
        merged = dict2.copy()  # Start with a copy of the first dictionary
        for key, value in dict1.items():
            if key in dict2:
                merged[key] = [value, dict2[key]]  # Combine values if key exists in both
            else:
                merged[key] = value  # Add the new key-value pair
        return merged
    
    def to_power(self, dB):
        return 10**(dB/10)

    def to_dB(self, power):
        return 10*np.log10(power)
    
    def curve_fit_Cvv_Dvv(self, x_arr, y_arr):
        x_arr = np.array(x_arr)
        y_arr = np.array(y_arr)
        
        def exp_func(x, c, d):
            return c * np.log(d * x + 1e-9) #add a small value to avoid log(0)
        
        try:
            params, covariance = curve_fit(exp_func, x_arr, y_arr)
            Cvv, Dvv = params
            return [Cvv, Dvv]
        except:
            return [np.nan, np.nan]
    
    def residuals_local(self, params, vv_obs, theta_rad, vwc):
        A, B, mv, s = params
        ks = self.k * s
        V1, V2 = vwc, vwc

        # Oh et al. (2004) model
        o = Oh04(mv, ks, theta_rad)
        vh_soil, vv_soil, hh_soil = o.get_sim()

        # Water Cloud Model (WCM)
        vv_sim, _, _ = WCM(A, B, V1, V2, theta_rad, vv_soil)

        vv_residual = np.sqrt(np.square(vv_obs - vv_sim))

        return vv_residual
    
    def residuals_global(self, params, A, B, ks, vv_obs, theta_rad, vwc):
        mv = params
        V1, V2 = vwc, vwc

        # Oh et al. (2004) model
        o = Oh04(mv, ks, theta_rad)
        vh_soil, vv_soil, hh_soil = o.get_sim()

        # Water Cloud Model (WCM)
        vv_sim, _, _ = WCM(A, B, V1, V2, theta_rad, vv_soil)

        vv_residual = np.sqrt(np.square(vv_obs - vv_sim))

        return vv_residual
    
    def search_file(self, directory, extensions='csv', recursive=False, year_filter=None, station=None):

        if not os.path.isdir(directory):
            raise ValueError(f"Provided path is not a valid directory: {directory}")
    
        file_list = []

        # Walk through directory
        for root, _, files in os.walk(directory):
            for file in files:
                # Filter by extension
                if extensions and not file.lower().endswith(tuple(extensions)):
                    continue
                
                # Filter by year (or substring)
                if year_filter and str(year_filter) not in file:
                    continue

                # Filter by year (or substring)
                if station and str(station) not in file:
                    continue
                
                file_path = os.path.join(root, file)
                file_list.append(file_path)

            if not recursive:
                break
        
        return file_list

    def read_risma_bulk_csv(self, fname, S1_lot, depth):
        
        df = pd.read_csv(fname, header=None, low_memory=False)
        df = df.drop(index=[0,1,2,3,5])
        df = df.reset_index(drop=True)
        df[0] = pd.to_datetime(df[0], format='%Y-%m-%d %H:%M:%S')

        # Set the datetime column as the index
        df.set_index(df.columns[0], inplace=True)

        df.columns = df.iloc[0]
        df = df.iloc[1:]

        # remove first part of the columns name
        df.columns = [col.split('.')[1] for col in df.columns]

        # Keep rows around -1 and +1 06:00 pm based on index in the df
        df = df[(df.index.strftime('%H:%M') >= f'{S1_lot - 1}:00') & (df.index.strftime('%H:%M') <= f'{S1_lot + 1}:00')]

        # filter df's columns contain 'Soil water content'
        df = df[[col for col in df.columns if 'Soil water content' in col]]

        # filter df for different depth
        if depth == 0:
            df = df[[col for col in df.columns if '0 to 5 cm' in col]]
        else:
            df = df[[col for col in df.columns if f'{depth} cm' in col]]

        # Convert the soil moisture columns to numeric, handling non-numeric values
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')  # 'coerce' will set non-numeric values to NaN
        
        # create a dict of columns as key and reducer 'median' as value
        aggregate = dict(zip(df.columns, ['median', ] * len(df.columns)))
        
        # Resample the data by day, calculating mean for soil moisture and sum for precipitation
        daily_data = df.resample('D').agg(aggregate)

        # Melt the DataFrame to long format for plotting, including precipitation
        df_melted = daily_data.melt(ignore_index=False, var_name='sensor', value_name='value')
        df_melted.index.name = 'date'

        # add new column named doy
        df_melted['doy'] = df_melted.index.dayofyear

        # reset index inplace and set doy as index
        df_melted.reset_index(inplace=True)
        df_melted.set_index('date', inplace=True)

        # fill nan
        df_melted.ffill(inplace=True)
        df_melted.bfill(inplace=True)

        # drop Date inplace
        # df_melted.drop('Date', axis=1, inplace=True)

        return df_melted

    def read_radar_backscatter(self, fname, croptype):

        df = pd.read_csv(fname)

        # Replace 0.0 with np.nan
        df = df.replace(0.0, np.nan)

        # keep rows where croptype is equal 158 in df
        df = df[df['croptype'].isin([croptype, ])]

        return df
    
    def cal_wcm_veg_param(self, df_sigma, df_risma, ssm_inv_thr, default_wcm_params, ssr_lt_36deg, ssr_gt_36deg):

        wcm_param_doy = {}

        for lc, df_cluster in df_sigma.groupby('croptype'):

            # drop unnecessary columns
            df_cluster = df_cluster.drop(['system:index', '.geo', 'croptype'], axis=1)
            df_t = df_cluster.T
            df_t.index.rename('date', inplace=True)
            df_t.reset_index(inplace=True)

            # Add new column named 'band' by using of _** from20200902T001505_VH
            df_t['band'] = df_t['date'].apply(lambda x: x.split('_')[1])

            # convert date to datetime by removing _** from20200902T001505_VH
            df_t['date'] = pd.to_datetime(df_t['date'].apply(lambda x: x.split('_')[0][:8])).dt.strftime('%Y%m%d')

            # Calculate mean of duplicate rows in df_t on numeric columns
            df_t = df_t.groupby(['date', 'band']).mean().reset_index()

            # Assuming you want to group the DataFrame 'df' in groups of three rows:
            for g, df_c in df_t.groupby(np.arange(len(df_t)) // 4):

                # Transpose to have four columns per day
                df_ct = df_c.T

                # Get date
                date_string = list(set(df_ct.iloc[0].values))[0]
                date_object = datetime.strptime(date_string, '%Y%m%d')
                day_of_year = date_object.timetuple().tm_yday
                print(f'Croptype: {lc}, date: {date_object}, doy: {day_of_year}')

                # Use date as column names and drop first row
                df_ct.columns = df_ct.iloc[1]
                df_ct = df_ct[2:]
                df_ct.reset_index(inplace=True, drop=True)

                # Drop nodata
                df_ct = df_ct[df_ct != 0].dropna()

                if df_ct.shape[0] <= 30:
                    continue

                categorized_angle_Avv = defaultdict(list)
                categorized_angle_Bvv = defaultdict(list)
                categorized_angle_mvs = defaultdict(list)
                categorized_angle_ssr = defaultdict(list)
                categorized_angle_vv_soil = defaultdict(list)

                for idx, row in df_ct.iterrows():

                    vh, vv, angle, vwc = row.values

                    nearest_int_angle = round(angle)  # Find the nearest integer
                    
                    if not default_wcm_params:

                        ssm = df_risma[df_risma.doy == day_of_year].value.mean()

                        if angle < 36:
                            ssr = ssr_lt_36deg
                        else:
                            ssr = ssr_gt_36deg

                        A_init = 1
                        B_init = 0.25
                        
                    else:
                        wcm_params = default_wcm_params[day_of_year][nearest_int_angle]
                        A_init, B_init = wcm_params[0][0]
                        # Cvv, Dvv = wcm_params[0][1]
                        ssm, ssr = wcm_params[1]

                    ssm_max = ssm + ssm_inv_thr
                    ssm_min = ssm - ssm_inv_thr
                    if ssm_min < 0:
                        ssm_min = 0
                    
                    ssr_min = 0
                    ssr_max = 5

                    A_min = 0
                    A_max = 2

                    B_min = 0
                    B_max = 0.5

                    # Degrees to Rad
                    theta_rad0 = np.deg2rad(angle)

                    # Initial guess for mv and ks
                    initial_guess = [A_init, B_init, ssm, ssr]

                    # Perform the optimization
                    res = least_squares(self.residuals_local, initial_guess, args=(vv, theta_rad0, vwc), 
                        bounds=([A_min, B_min, ssm_min, ssr_min], [A_max, B_max, ssm_max, ssr_max]))
                    A, B, mv, s = res.x
                    ks = self.k * s

                    # Oh et al. (2004) model
                    o = Oh04(mv, ks, theta_rad0)
                    vh_soil, vv_soil, hh_soil = o.get_sim()

                    # # Water Cloud Model (WCM)
                    # V1, V2 = vwc, vwc
                    # vv_tot, vv_veg, tau = WCM(A_init, B_init, V1, V2, theta_rad0, vv_soil)

                    categorized_angle_Avv[nearest_int_angle].append(A)
                    categorized_angle_Bvv[nearest_int_angle].append(B)
                    categorized_angle_mvs[nearest_int_angle].append(mv)
                    categorized_angle_ssr[nearest_int_angle].append(s)
                    categorized_angle_vv_soil[nearest_int_angle].append(self.to_dB(vv_soil))

                categorized_angle_Avv_mean = dict(map(lambda el: (el[0], np.array(el[1]).mean()), categorized_angle_Avv.items()))
                categorized_angle_Bvv_mean = dict(map(lambda el: (el[0], np.array(el[1]).mean()), categorized_angle_Bvv.items()))
                categorized_angle_mvs_mean = dict(map(lambda el: (el[0], np.array(el[1]).mean()), categorized_angle_mvs.items()))
                categorized_angle_ssr_mean = dict(map(lambda el: (el[0], np.array(el[1]).mean()), categorized_angle_ssr.items()))
                merged_angle_Avv_Bvv = self.mergeDictionary(categorized_angle_Avv_mean, categorized_angle_Bvv_mean)
                merged_angle_mvs_ssr = self.mergeDictionary(categorized_angle_mvs_mean, categorized_angle_ssr_mean)

                merged_angle_vv_soils_mvs = self.mergeDictionary(categorized_angle_mvs, categorized_angle_vv_soil)
                merged_angle_Cvv_Dvv = dict(map(lambda el: (el[0], self.curve_fit_Cvv_Dvv(el[1][0], el[1][1])), 
                    merged_angle_vv_soils_mvs.items()))
                
                merged_wcm_params = self.mergeDictionary(merged_angle_Avv_Bvv, merged_angle_Cvv_Dvv)
                merged_wcm_params = self.mergeDictionary(merged_wcm_params, merged_angle_mvs_ssr)
                
                wcm_param_doy[day_of_year] = merged_wcm_params
        
        return wcm_param_doy

    def cal_wcm_soil_param(self, df_sigma, wcm_veg_param_df):
        wcm_param_doy = {}

        for lc, df_cluster in df_sigma.groupby('croptype'):

            # drop unnecessary columns
            df_cluster = df_cluster.drop(['system:index', '.geo', 'croptype'], axis=1)
            df_t = df_cluster.T
            df_t.index.rename('date', inplace=True)
            df_t.reset_index(inplace=True)

            # Add new column named 'band' by using of _** from20200902T001505_VH
            df_t['band'] = df_t['date'].apply(lambda x: x.split('_')[1])

            # convert date to datetime by removing _** from20200902T001505_VH
            df_t['date'] = pd.to_datetime(df_t['date'].apply(lambda x: x.split('_')[0][:8])).dt.strftime('%Y%m%d')

            # Calculate mean of duplicate rows in df_t on numeric columns
            df_t = df_t.groupby(['date', 'band']).mean().reset_index()

            # Assuming you want to group the DataFrame 'df' in groups of three rows:
            for g, df_c in df_t.groupby(np.arange(len(df_t)) // 4):

                # Transpose to have four columns per day
                df_ct = df_c.T

                # Get date
                date_string = list(set(df_ct.iloc[0].values))[0]
                date_object = datetime.strptime(date_string, '%Y%m%d')
                day_of_year = date_object.timetuple().tm_yday
                print(f'Croptype: {lc}, date: {date_object}, doy: {day_of_year}')

                # Use date as column names and drop first row
                df_ct.columns = df_ct.iloc[1]
                df_ct = df_ct[2:]
                df_ct.reset_index(inplace=True, drop=True)

                # Drop nodata
                df_ct = df_ct[df_ct != 0].dropna()

                if df_ct.shape[0] <= 30:
                    continue
                
                categorized_angle_mvs = defaultdict(list)
                categorized_angle_vv_soil = defaultdict(list)

                for idx, row in df_ct.iterrows():
                    vh, vv, angle, vwc = row.values
                    
                    # Find the nearest integer
                    nearest_int_angle = round(angle)
                    # print(wcm_veg_param_doy[day_of_year][nearest_int_angle])

                    # extract Avv and Bvv
                    Avv = wcm_veg_param_df[(wcm_veg_param_df.doy == day_of_year) & (wcm_veg_param_df.angle == nearest_int_angle)].Avv.values[0]
                    Bvv = wcm_veg_param_df[(wcm_veg_param_df.doy == day_of_year) & (wcm_veg_param_df.angle == nearest_int_angle)].Bvv.values[0]
                    ssr = wcm_veg_param_df[(wcm_veg_param_df.doy == day_of_year) & (wcm_veg_param_df.angle == nearest_int_angle)].ssr.values[0]
                    # Avv = wcm_veg_param_doy[day_of_year][nearest_int_angle][0][0][0]
                    # Bvv = wcm_veg_param_doy[day_of_year][nearest_int_angle][0][0][1]
                    # ssr = wcm_veg_param_doy[day_of_year][nearest_int_angle][1][1]
                    # print(Avv, Bvv, ssr)
                    
                    # Degrees to Rad
                    theta_rad0 = np.deg2rad(angle)

                    # Perform the optimization
                    ks = self.k * ssr
                    res = differential_evolution(self.residuals_global, bounds=[(0.01, 0.75), ], args=(Avv, Bvv, ks, vv, theta_rad0, vwc))
                    mv = res.x[0]

                    # Oh et al. (2004) model
                    o = Oh04(mv, ks, theta_rad0)
                    vh_soil, vv_soil, hh_soil = o.get_sim()
                    # print(mv, vh_soil, vv_soil, hh_soil)

                    categorized_angle_mvs[nearest_int_angle].append(mv)
                    categorized_angle_vv_soil[nearest_int_angle].append(self.to_dB(vv_soil))

                categorized_angle_mvs_mean = dict(map(lambda el: (el[0], np.array(el[1]).mean()), categorized_angle_mvs.items()))

                merged_angle_vv_soils_mvs = self.mergeDictionary(categorized_angle_mvs, categorized_angle_vv_soil)
                merged_angle_Cvv_Dvv = dict(map(lambda el: (el[0], self.curve_fit_Cvv_Dvv(el[1][0], el[1][1])), 
                    merged_angle_vv_soils_mvs.items()))
                
                merged_wcm_params = self.mergeDictionary(merged_angle_Cvv_Dvv, categorized_angle_mvs_mean)
                
                wcm_param_doy[day_of_year] = merged_wcm_params
        
        return wcm_param_doy