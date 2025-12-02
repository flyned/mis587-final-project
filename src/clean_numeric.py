import numpy as np
import re


def parse_rent(val):
    if "Not Disclosed" in val:
        return np.nan
    val = val.replace("(Est.)", "").replace("$", "").strip()
    if "-" in val:
        low, high = val.split("-")
        try:
            return (float(low) + float(high)) / 2
        except:
            return np.nan
    else:
        try:
            return float(val)
        except:
            return np.nan

def to_feet(val):
    if not isinstance(val, str) or val.strip() == "":
        return np.nan
    match = re.match(r"(\d+)'(\d+)\"", val.strip())
    if not match:
        return np.nan
    feet, inches = map(int, match.groups())
    return feet + inches / 12

def parse_cranes(val):
    try:
        return int(val)
    except ValueError:
        return 0

def parse_drive_ins(val):
    if not isinstance(val, str) or val.strip() == "" or val.strip().lower() == "none":
        return np.nan
    # If the string starts with a number followed by "/", capture that number
    match = re.match(r"(\d+)", val.strip())
    if match:
        return int(match.group(1))
    return np.nan

def parse_expenses(val):
    if not isinstance(val, str) or val.strip() == "":
        return np.nan
    match = re.search(r"\$([0-9.]+)/sf", val)
    if match:
        return float(match.group(1))
    return np.nan


def clean_numeric(df):
    df["Rent/SF/Yr"] = df["Rent/SF/Yr"].apply(parse_rent)
    df["Ceiling Ht"] = df["Ceiling Ht"].apply(to_feet)
    df["Number Of Cranes"] = df["Number Of Cranes"].apply(parse_cranes)
    df["Drive Ins"] = df["Drive Ins"].apply(parse_drive_ins)
    df["Building Operating Expenses"] = df["Building Operating Expenses"].apply(parse_expenses)
    df["Building Tax Expenses"] = df["Building Tax Expenses"].apply(parse_expenses)
