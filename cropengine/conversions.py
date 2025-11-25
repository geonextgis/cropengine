import numpy as np


def K_to_C(series):
    return series - 273.15


def J_to_kJ(series):
    return series / 1000


def m_to_cm(series):
    return series * 100


def m_to_mm(series):
    return series * 1000


def none(series):
    return series


def Td_to_VAP(dewpoint_K):
    """Compute vapour pressure (hPa) from dewpoint temperature (K)."""
    # Handle Series or scalar
    Td_C = dewpoint_K - 273.15
    vap_kPa = 0.6108 * np.exp((17.27 * Td_C) / (Td_C + 237.3))
    return vap_kPa


def uv_to_wind(u, v):
    """Compute wind speed magnitude (m/s)."""
    return np.sqrt(u**2 + v**2)


# Registry for easy lookup
CONVERSION_FUNCS = {
    "none": none,
    "K_to_C": K_to_C,
    "J_to_kJ": J_to_kJ,
    "m_to_cm": m_to_cm,
    "m_to_mm": m_to_mm,
    "Td_to_VAP": Td_to_VAP,
    "uv_to_wind": uv_to_wind,
}
