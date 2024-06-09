import pandas as pd
import pickle

from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
from typing import Literal, Tuple


PARTITION = "partition_1"
OUTLIER_FLAG = "outlier_flag_1"
SAMPLE_ORDER = "sample_order_1"
WAVELENGTH_RANGES = {
    "range_684to990": {
        "min": 684,
        "max": 990
    },
    "range_720to990": {
        "min": 720,
        "max": 990
    },
    "range_600to990": {
        "min": 600,
        "max": 990
    },
    "range_402to990": {
        "min": 402,
        "max": 990
    },
}


def get_preprocess_data(study_name: str, return_partitioned: bool = True) -> Tuple[pd.DataFrame, list, list]:
    df = pd.read_pickle(f"data/{study_name}/preprocessed_data/dataset.pkl")
    with open(f"data/{study_name}/preprocessed_data/columns.pkl", "rb") as file:
        y_col, x_cols = pickle.load(file)
    
    # split into calibration and tuning sets 
    if return_partitioned:
        df_cal = df.query("partition == 'train' and train_partition == 'calibration'")
        df_tune = df.query("partition == 'train' and train_partition == 'tunning'")
        return df_cal, df_tune, y_col, x_cols
    else:
        return df, y_col, x_cols


def preprocess_data(
        wavelength_range: Literal["range_684to990", "range_720to990", "range_600to990", "range_402to990"], 
        pretreatment: Literal["pretreatment_0", "pretreatment_1", "pretreatment_2", "pretreatment_3"],
        savgol_window_size: int = None,
        return_partitioned: bool = True
    ) -> Tuple[pd.DataFrame, list, list]:
    
    # read in prepared dataset without spectra preprocessing
    df = pd.read_pickle("data/input/mango_dmc_and_spectra_v2.pkl")
    wavelength_cols = df.filter(regex="^\d+", axis=1).columns
    descriptive_cols = [col for col in df.columns if col not in wavelength_cols]
    wavelength_cols = wavelength_cols.astype(int).tolist()
    df.columns = descriptive_cols + wavelength_cols
    
    # apply pretreatment to spectra
    initial_wavelengths = [w for w in wavelength_cols if 309 <= w <= 1149]
    pretreatments = apply_pretreatment(pretreatment, spectra=df[wavelength_cols], wavelengths=initial_wavelengths, savgol_window_size=savgol_window_size)
    
    # trim wavelength range
    w_range = WAVELENGTH_RANGES[wavelength_range]
    selected_wavelength_cols = [w for w in wavelength_cols if w_range["min"] <= w <= w_range["max"]]
    
    # compile pretreatment with selected wavelength range into dataframe with metadata
    objs = [df[descriptive_cols]]
    for pretreatment in pretreatments:
        objs.append(
            pretreatment["data"][selected_wavelength_cols].add_suffix(f"_{pretreatment['name']}")
        )    
    df = pd.concat(
        objs=objs,
        axis=1
    )
    
    # prepare dataset based on specified order and partition 
    df.sort_values(by=SAMPLE_ORDER, inplace=True)
    df.insert(loc=1, column="sample_order", value=df[SAMPLE_ORDER])
    df.insert(loc=0, column="partition", value=df[PARTITION])
    df.insert(loc=1, column="train_partition", value=df[f"train_{PARTITION}"])
    df.drop(columns=[col for col in df.columns if col.startswith("partition_") and col != "partition_ext"], inplace=True)
    df.drop(columns=[col for col in df.columns if col.startswith("train_partition_")], inplace=True)
    df.drop(columns=[col for col in df.columns if col.startswith("sample_order_")], inplace=True)
    
    # remove outliers based on specified outlier flag
    df = df[df[OUTLIER_FLAG] == 0].copy()
    df = df.query("subsequent_flag_1 == 0").copy()
    df.drop(columns=[col for col in df.columns if col.startswith("outlier_flag_")], inplace=True)
    
    # get the x, y and descriptive columns
    x_cols = df.filter(regex="^\d+", axis=1).columns.tolist()
    y_col = ["dry_matter"]
    descriptive_cols = [col for col in df.columns if col not in x_cols]
    
    # apply standard scalar 
    scaler = StandardScaler()
    scaler.fit(X=df.query("partition in ('train', 'validation')")[x_cols])  # train scaler using the entire training set
    df[x_cols] = scaler.transform(df[x_cols])
    
    # split into calibration and tuning sets 
    if return_partitioned:
        df_cal = df.query("partition == 'train' and train_partition == 'calibration'")
        df_tune = df.query("partition == 'train' and train_partition == 'tunning'")
        return df_cal, df_tune, y_col, x_cols
    else:
        return df, y_col, x_cols


def apply_pretreatment(pretreatment: str, spectra: pd.DataFrame, wavelengths: list, savgol_window_size: int = None) -> list:
    if pretreatment == "pretreatment_0":
        pretreatments = [{"name": "abs", "data": spectra[wavelengths]}]
    elif pretreatment == "pretreatment_1":
        pretreatments = apply_pretreatment_1(spectra, wavelengths, savgol_window_size)
    elif pretreatment == "pretreatment_2":
        pretreatments = apply_pretreatment_2(spectra, wavelengths, savgol_window_size)
    elif pretreatment == "pretreatment_3":
        pretreatments = apply_pretreatment_3(spectra, wavelengths, savgol_window_size)
    else:
        raise ValueError(f"{pretreatment} has not been configured")
    return pretreatments


def apply_pretreatment_1(spectra: pd.DataFrame, wavelengths: list, savgol_window_size: int) -> list:
    
    # Savitzky-Golay smoothing, 2nd deriv
    p1 = spectra[wavelengths].apply(
        lambda row: savgol_filter(
            x=row,
            window_length=savgol_window_size,
            polyorder=2,
            deriv=2
        ),
        axis=1,
        result_type="expand"
    )
    p1.columns = wavelengths
    
    return [{
        "name": "savgol_d2",
        "data": p1
    }]


def apply_pretreatment_2(spectra: pd.DataFrame, wavelengths: list, savgol_window_size: int) -> list:
    
    pretreatments = []
    
    # raw absorbance
    p1 = spectra[wavelengths]
    pretreatments.append({"name": "abs", "data": p1})
    
    # standard normal variate (SNV)
    p2 = spectra[wavelengths].apply(lambda x: (x - x.mean()) / x.std())
    pretreatments.append({"name": "snv", "data": p2})
    
    # Savitzky-Golay smoothing, 1st deriv
    p3 = spectra[wavelengths].apply(
        lambda row: savgol_filter(
            x=row,
            window_length=savgol_window_size,
            polyorder=2,
            deriv=1
        ),
        axis=1,
        result_type="expand"
    )
    p3.columns = wavelengths
    pretreatments.append({"name": "savgol_d1", "data": p3})
    
    # Savitzky-Golay smoothing, 2nd deriv
    p4 = spectra[wavelengths].apply(
        lambda row: savgol_filter(
            x=row,
            window_length=savgol_window_size,
            polyorder=2,
            deriv=2
        ),
        axis=1,
        result_type="expand"
    )
    p4.columns = wavelengths
    pretreatments.append({"name": "savgol_d2", "data": p4})
    
    # SNV + Savitzky-Golay smoothing (1st deriv)
    p5 = p2.apply(
        lambda row: savgol_filter(
            x=row,
            window_length=savgol_window_size,
            polyorder=2,
            deriv=1
        ),
        axis=1,
        result_type="expand"
    )
    p5.columns = wavelengths
    pretreatments.append({"name": "snv_savgol_d1", "data": p5})
    
    # SNV + Savitzky-Golay smoothing (2nd deriv)
    p6 = p2.apply(
        lambda row: savgol_filter(
            x=row,
            window_length=savgol_window_size,
            polyorder=2,
            deriv=2
        ),
        axis=1,
        result_type="expand"
    )
    p6.columns = wavelengths
    pretreatments.append({"name": "snv_savgol_d2", "data": p6})
    
    return pretreatments


def apply_pretreatment_3(spectra: pd.DataFrame, wavelengths: list, savgol_window_size: int) -> list:
    
    pretreatments = []
    
    # standard normal variate (SNV)
    p1 = spectra[wavelengths].apply(lambda x: (x - x.mean()) / x.std())
    
    # SNV + Savitzky-Golay smoothing (2nd deriv)
    p2 = p1.apply(
        lambda row: savgol_filter(
            x=row,
            window_length=savgol_window_size,
            polyorder=2,
            deriv=2
        ),
        axis=1,
        result_type="expand"
    )
    p2.columns = wavelengths
    pretreatments.append({"name": "snv_savgol_d2", "data": p2})
    
    return pretreatments