#!/usr/bin/env python3
"""
plot_braking_behavior.py — visualize braking behavior from trajectories.braking_events,
and optionally overlay a heatmap on an orthophoto TIFF.

This script assumes you have migrated to *instantaneous peak braking*:
  - braking_events.a_min exists (most negative accel during the event, m/s^2)
  - braking strength is |a_min| (positive magnitude)

If your table does not have a_min, fix the table + ingestion first. This script will
NOT fall back to avg_decel.
"""

import os
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl  # add once at top of file
import matplotlib.colors as mcolors


from stat_analysis import (
    quantify_periods,
    frequency_rate_table,
    hourly_frequency_table,
    add_local_hour,
)


from dotenv import load_dotenv
load_dotenv(override=False)  # loads .env from current working dir


INTERSECTION_META = {
    "1": {
        "tex_name": r"Overseas Hwy / Roosevelt Blvd",
        "prefix": "overseas_roosevelt",
        "label": "fig:hourlyevents_overseas_roosevelt",
    },
    "2": {
        "tex_name": r"Truman Avenue / White Street",
        "prefix": "truman_white",
        "label": "fig:hourlyevents_truman_white",
    },
}


# NOTE: braking.chio already handles making gui/ importable (repo layout),
# so we avoid repeating sys.path hacks here.
from braking.chio import ch_query_json_each_row, ClickHouseHTTP


plt.rcParams.update({
    "font.size": 14,          # base font size
    "axes.titlesize": 18,     # title
    "axes.labelsize": 16,     # x/y labels
    "xtick.labelsize": 14,    # x tick labels
    "ytick.labelsize": 14,    # y tick labels
    "legend.fontsize": 14,
})


def esc(s: str) -> str:
    """Tiny SQL escaper for single quotes (used because we format SQL strings)."""
    return s.replace("'", "\\'")


def hour_to_ampm(h: int) -> str:
    """0-23 -> '12 AM', '1 AM', ..., '12 PM', '1 PM', ..."""
    h = int(h)
    suffix = "AM" if h < 12 else "PM"
    h12 = h % 12
    if h12 == 0:
        h12 = 12
    return f"{h12} {suffix}"


def fetch_events_df(
    ch: ClickHouseHTTP,
    intersection_id: str,
    approach_id: str,
    video: Optional[str] = None,
    since: Optional[str] = None,
) -> pd.DataFrame:
    """
    Pull braking events (no positions). Requires a_min.
    """
    db = ch.db
    where = [
        f"intersection_id = '{esc(intersection_id)}'",
        f"approach_id = '{esc(approach_id)}'",
    ]
    if video:
        where.append(f"video = '{esc(video)}'")
    if since:
        where.append(f"created_at >= toDateTime('{esc(since)}')")

    where_sql = " AND ".join(where)

    sql = f"""
    SELECT
        intersection_id,
        approach_id,
        video,
        track_id,
        class,
        t_start, t_end,
        r_start, r_end,
        v_start, v_end,
        dv,
        a_min,
        avg_decel,
        severity,
        event_ts,
        created_at
    FROM {db}.braking_events
    WHERE {where_sql}
    ORDER BY created_at ASC
    FORMAT JSONEachRow
    """

    rows = ch_query_json_each_row(ch, sql)
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # timestamps
    df["event_ts"] = pd.to_datetime(df.get("event_ts"), errors="coerce")
    df["created_at"] = pd.to_datetime(df.get("created_at"), errors="coerce", utc=True)

    # numerics
    for c in ["t_start", "t_end", "r_start", "r_end", "v_start", "v_end", "dv", "a_min", "avg_decel"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if "a_min" not in df.columns:
        raise RuntimeError("braking_events.a_min is missing. Stop and fix schema/ingestion.")

    # braking strength magnitude (positive)
    df["decel_mag"] = -df["a_min"]

    # sanity: decel_mag must be finite and >= 0
    df.loc[~np.isfinite(df["decel_mag"]), "decel_mag"] = np.nan
    df.loc[df["decel_mag"] < 0, "decel_mag"] = np.nan

    return df


def print_hourly_stacked_latex(
    df: pd.DataFrame,
    intersection_id: str,
    prefix: str,
) -> None:
    """
    Print LaTeX snippet for the hourly stacked-bar braking figure.

    Uses INTERSECTION_META for nice intersection name, prefix, and label.
    """
    meta = INTERSECTION_META.get(str(intersection_id), {})
    tex_name = meta.get("tex_name", f"Intersection {intersection_id}")
    label = meta.get("label", f"fig:hourlyevents_{prefix}")

    # how many unique local days in df?
    if df is None or df.empty:
        days_total = 0
    else:
        d = add_local_hour(df.copy())
        days_total = int(pd.Series(d["date_local"].dropna()).nunique())

    days_str = (
        f"{days_total} day" if days_total == 1
        else f"{days_total} days"
        if days_total > 0 else "multiple days"
    )

    print()
    print(r"\begin{figure*}[tb]")
    print(r"    \centering")
    print(
        rf"    \includegraphics[width=0.9\linewidth]"
        rf"{{{{figures/{prefix}_hourly_event_severity_stacked.pdf}}}}"
    )
    print(
        rf"    \caption{{Average hourly braking event counts by severity "
        rf"(mild, moderate, severe) over {days_str} of traffic footage at {tex_name}.}}"
    )
    print(rf"    \label{{{label}}}")
    print(r"\end{figure*}")
    print()


def fetch_event_positions_df(
    ch: ClickHouseHTTP,
    intersection_id: str,
    approach_id: str,
    video: Optional[str] = None,
    since: Optional[str] = None,
) -> pd.DataFrame:
    """
    Join braking_events -> raw points within each event window, then aggregate to a
    representative (x_m, y_m) per braking event (for heatmapping).

    Requires a_min.
    """
    db = ch.db

    e_where = [
        f"e.intersection_id = '{esc(intersection_id)}'",
        f"e.approach_id = '{esc(approach_id)}'",
    ]
    if video:
        e_where.append(f"e.video = '{esc(video)}'")
    if since:
        e_where.append(f"e.created_at >= toDateTime('{esc(since)}')")

    e_where_sql = " AND ".join(e_where)

    sql = f"""
    SELECT
        e.intersection_id,
        e.approach_id,
        e.video,
        e.track_id,
        e.t_start,
        e.t_end,
        e.a_min,
        e.dv,
        e.severity,

        avg(r.map_px_x) AS col_px,
        avg(r.map_px_y) AS row_px

    FROM {db}.braking_events AS e
    INNER JOIN {db}.raw AS r
        ON r.video = e.video
        AND r.track_id = e.track_id
        AND r.intersection_id = toUInt8(e.intersection_id)

    WHERE {e_where_sql}
    AND r.secs >= e.t_start
    AND r.secs <= e.t_end
    AND (r.map_px_x != 0 OR r.map_px_y != 0)
    GROUP BY
        e.intersection_id, e.approach_id, e.video,
        e.track_id, e.t_start, e.t_end, e.a_min, e.dv, e.severity
    FORMAT JSONEachRow
    """


    rows = ch_query_json_each_row(ch, sql)
    if not rows:
        return pd.DataFrame()

    dfp = pd.DataFrame(rows)
    for c in ["a_min", "dv", "col_px", "row_px"]:
        dfp[c] = pd.to_numeric(dfp[c], errors="coerce")

    if "a_min" not in dfp.columns:
        raise RuntimeError("braking_events.a_min is missing in join output. Fix schema/ingestion.")

    dfp["decel_mag"] = -dfp["a_min"]
    dfp.loc[~np.isfinite(dfp["decel_mag"]), "decel_mag"] = np.nan
    dfp.loc[dfp["decel_mag"] < 0, "decel_mag"] = np.nan
    return dfp


def _savefig(fig, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Only call tight_layout if figure is NOT using constrained_layout
    try:
        if not getattr(fig, "get_constrained_layout", lambda: False)():
            fig.tight_layout()
    except Exception:
        # If anything goes wrong, skip layout adjustment
        pass

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)



def plot_dashboard(df: pd.DataFrame, out_dir: Path, title: str = "", prefix: str = "braking") -> None:
    """
    Saves a small dashboard (multiple pdfs) using peak decel magnitude |a_min|.
    """
    if df.empty:
        print("No events to plot.")
        return

    df = df[np.isfinite(df["decel_mag"]) & np.isfinite(df["dv"])]
    if df.empty:
        print("No finite events to plot.")
        return

    decel = df["decel_mag"].values
    dv = df["dv"].values

    sev_order = ["mild", "moderate", "severe"]
    sev_counts = [int((df["severity"] == s).sum()) for s in sev_order]

    ylab = "|a_min|\n(m/s²)  [peak decel]"

    # ---- Severity counts ----
    fig = plt.figure()
    plt.bar(sev_order, sev_counts)
    plt.ylabel("Count")
    plt.title(f"Braking severity counts {title}")
    plt.grid(axis="y", alpha=0.3)
    _savefig(fig, out_dir / f"{prefix}_severity_counts.pdf")

    # ---- Histogram of braking strength (peak) ----
    fig = plt.figure()
    bins = np.linspace(0, max(4.0, np.nanpercentile(decel, 99)), 40)
    plt.hist(decel, bins=bins)
    plt.xlabel(ylab)
    plt.ylabel("Events")
    plt.title(f"Distribution of peak braking strength {title}")
    plt.grid(alpha=0.3)
    _savefig(fig, out_dir / f"{prefix}_peak_decel_hist.pdf")

    # ---- CDF of braking strength (peak) ----
    fig = plt.figure()
    xs = np.sort(decel)
    ys = np.arange(1, len(xs) + 1) / len(xs)
    plt.plot(xs, ys)
    plt.xlabel(ylab)
    plt.ylabel("CDF")
    plt.title(f"CDF of peak braking strength {title}")
    plt.grid(alpha=0.3)
    _savefig(fig, out_dir / f"{prefix}_peak_decel_cdf.pdf")

    # ---- Histogram of dv ----
    fig = plt.figure()
    bins_dv = np.linspace(0, max(3.0, np.nanpercentile(dv, 99)), 40)
    plt.hist(dv, bins=bins_dv)
    plt.xlabel("Δv over event (m/s)")
    plt.ylabel("Events")
    plt.title(f"Distribution of speed drops {title}")
    plt.grid(alpha=0.3)
    _savefig(fig, out_dir / f"{prefix}_dv_hist.pdf")

    # ---- Scatter: dv vs peak braking strength ----
    fig = plt.figure()
    plt.scatter(decel, dv, s=8, alpha=0.4)
    plt.xlabel(ylab)
    plt.ylabel("Δv (m/s)")
    plt.title(f"Peak braking vs total slowdown {title}")
    plt.grid(alpha=0.3)
    _savefig(fig, out_dir / f"{prefix}_dv_vs_peak_decel.pdf")

    # ---- per-hour-of-day boxplots of peak braking ----
    time_col = None
    if "event_ts" in df.columns and df["event_ts"].notna().any():
        time_col = "event_ts"
    elif "created_at" in df.columns:
        time_col = "created_at"

    if time_col is not None:
        ca = pd.to_datetime(df[time_col], errors="coerce")  # NOTE: no utc=True

        # If timestamps are tz-naive, assume they are ALREADY local ET.
        # If they're tz-aware (e.g., stored UTC), convert to ET.
        try:
            if ca.dt.tz is None:
                ca_local = ca.dt.tz_localize(
                    "America/New_York",
                    nonexistent="shift_forward",
                    ambiguous="NaT",
                )
            else:
                ca_local = ca.dt.tz_convert("America/New_York")
        except Exception:
            ca_local = ca  # last-resort fallback

        dfh = df.copy()
        dfh["hour"] = ca_local.dt.hour
        dfh = dfh.dropna(subset=["hour", "decel_mag"])
        dfh = dfh[dfh["hour"] != 19]  # remove 7 PM

        if not dfh.empty:
            hours_present = sorted(dfh["hour"].unique().astype(int).tolist())
            data = [dfh.loc[dfh["hour"] == h, "decel_mag"].values for h in hours_present]
            labels = [hour_to_ampm(h) for h in hours_present]

            fig = plt.figure(figsize=(12, 4))
            plt.boxplot(data, labels=labels, showfliers=False)
            plt.xlabel("Hour of day")
            plt.xticks(rotation=45, ha="right")
            plt.ylabel(ylab)
            print(f"Peak braking strength by hour of day {title}")
            plt.grid(axis="y", alpha=0.3)
            _savefig(fig, out_dir / f"{prefix}_hourly_peak_boxplot.pdf")

        # ---- per-hour-of-day boxplots of avg_decel ----
        if "avg_decel" in dfh.columns:
            dfh2 = dfh.dropna(subset=["hour", "avg_decel"])
            if not dfh2.empty:
                hours_present2 = sorted(dfh2["hour"].unique().astype(int).tolist())
                data2 = [dfh2.loc[dfh2["hour"] == h, "avg_decel"].values for h in hours_present2]
                labels2 = [hour_to_ampm(h) for h in hours_present2]

                fig = plt.figure(figsize=(12, 4))
                plt.boxplot(data2, labels=labels2, showfliers=False)
                plt.xlabel("Hour of day")
                plt.xticks(rotation=45, ha="right")
                plt.ylabel("avg_decel (m/s²)\n[Δv/dt over event]")
                print(f"Average braking decel by hour of day {title}")
                plt.grid(axis="y", alpha=0.3)
                _savefig(fig, out_dir / f"{prefix}_hourly_avg_decel_boxplot.pdf")

    print(f"Saved dashboard pdfs to: {out_dir}")

US_SURVEY_FT_TO_M = 0.30480060960121924


def plot_heatmap_on_tiff(dfp, tiff_path, out_path, bins=250, weight_mode="count"):
    if dfp.empty:
        print("No positioned events for heatmap.")
        return

    import rasterio

    cols = dfp["col_px"].to_numpy()
    rows = dfp["row_px"].to_numpy()

    w = None
    if weight_mode == "peak":
        w = dfp["decel_mag"].to_numpy()

    with rasterio.open(tiff_path) as src:
        img = src.read()
        H, W = img.shape[1], img.shape[2]

        inside = (cols >= 0) & (cols < W) & (rows >= 0) & (rows < H)
        cols = cols[inside]
        rows = rows[inside]
        if w is not None:
            w = w[inside]

        if cols.size == 0:
            print("All heatmap points fell outside TIFF bounds.")
            return

        heat, _, _ = np.histogram2d(
            cols, rows,
            bins=bins,
            range=[[0, W], [0, H]],
            weights=w,
        )
        heat = heat.T
        heat_norm = heat / heat.max() if heat.max() > 0 else heat

        bg = np.transpose(img[:3], (1, 2, 0)) if img.shape[0] >= 3 else img[0]

    # --- main heatmap (full ortho) ---
    fig = plt.figure(figsize=(10, 8))
    plt.imshow(bg)
    plt.imshow(heat_norm, alpha=0.45, cmap="hot")
    plt.title(f"Braking heatmap overlay (weight={weight_mode})")
    plt.axis("off")
    _savefig(fig, out_path)
    print(f"Saved heatmap overlay to: {out_path}")

    # --- DEBUG: zoomed scatter of raw points around their bounding box ---
    debug_out = out_path.with_name(out_path.stem + "_scatter.pdf")
    fig_dbg = plt.figure(figsize=(10, 8))
    ax = plt.gca()
    ax.imshow(bg)

    # severities corresponding to the filtered cols/rows
    sev_all = dfp["severity"].to_numpy()
    sev = sev_all[inside]

    # Define an order + colors for severity
    severity_order = ["mild", "moderate", "severe"]
    severity_colors = {
        "mild":    "#32a852",   # green-ish
        "moderate":"#f9a11b",   # orange-ish
        "severe":  "#d62728",   # red-ish
    }


    # Plot each severity separately so we can color + legend
    for s in severity_order:
        mask_s = (sev == s)
        if not np.any(mask_s):
            continue
        ax.scatter(
            cols[mask_s],
            rows[mask_s],
            s=25,
            alpha=0.7,
            label=s.capitalize(),
            color=severity_colors.get(s, "tab:gray"),
        )

    # Compute tight bounds around points and add padding
    x_min, x_max = float(cols.min()), float(cols.max())
    y_min, y_max = float(rows.min()), float(rows.max())

    span_x = x_max - x_min
    span_y = y_max - y_min

    pad_x = max(40.0, 0.80 * span_x)
    pad_y = max(40.0, 0.80 * span_y)

    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)

    # Final window, clamped to image bounds
    x_lo = max(0.0, cx - (span_x / 2 + pad_x))
    x_hi = min(float(W), cx + (span_x / 2 + pad_x))
    y_lo = max(0.0, cy - (span_y / 2 + pad_y))
    y_hi = min(float(H), cy + (span_y / 2 + pad_y))

    # --- NEW: crop vertically from both top *and* bottom ---
    # crop_factor < 1.0 means "keep only this fraction of the height"
    crop_factor = 1.0  # try 0.5–0.8 to taste
    span_y_window = y_hi - y_lo
    half_new = 0.5 * crop_factor * span_y_window
    cy_window = 0.5 * (y_lo + y_hi)

    y_lo = max(0.0, cy_window - half_new)
    y_hi = min(float(H), cy_window + half_new)

    # origin='upper' for imshow → y axis is inverted (max at top)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_hi, y_lo)

    ax.set_title("Braking event positions (zoomed col_px,row_px)")
    ax.axis("off")
    ax.legend(loc="upper right", frameon=True, fontsize=8)

    _savefig(fig_dbg, debug_out)
    print(f"Saved zoomed debug scatter overlay to: {debug_out}")



def plot_frequency_rates(df: pd.DataFrame, out_dir: Path, title: str = "", prefix: str = "braking") -> None:
    tab = frequency_rate_table(df, restrict_to_hours_7_to_18=True, alpha=0.05)
    if tab.empty:
        print("No frequency data to plot.")
        return

    label_map = {"AM_peak": "7–9 AM", "Midday": "11 AM–1 PM", "PM_peak": "4–6 PM"}

    tab = tab.copy()
    tab["period"] = pd.Categorical(tab["period"], categories=["AM_peak", "Midday", "PM_peak"], ordered=True)
    tab = tab.sort_values("period")

    x_labels = [label_map.get(p, str(p)) for p in tab["period"].astype(str).tolist()]
    rates = tab["rate_per_hr"].to_numpy()
    lo = tab["ci_lo"].to_numpy()
    hi = tab["ci_hi"].to_numpy()
    counts = tab["count"].to_numpy()
    yerr = np.vstack([rates - lo, hi - rates])

    # ✅ local font sizes for small single-column plot
    with mpl.rc_context({
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    }):
        fig = plt.figure(figsize=(3, 2.15), constrained_layout=True)
        x = np.arange(len(x_labels))
        bar_width = 0.55

        plt.bar(x, rates, width=bar_width)
        plt.errorbar(x, rates, yerr=yerr, fmt="none", capsize=2.5, linewidth=0.9)

        plt.title("Braking event rate by period", pad=2)
        plt.ylabel("Events / hour")
        plt.xticks(x, x_labels, rotation=15, ha="right")


        plt.grid(axis="y", alpha=0.3)

        # n labels: smaller + lower
        for i in range(len(x)):
            y_txt = rates[i] * 0.88
            plt.text(i, y_txt, f"n={int(counts[i])}", ha="center", va="center", fontsize=7, color="white")

        out_path = out_dir / f"{prefix}_frequency_rates.pdf"
        _savefig(fig, out_path)
        print(f"Saved frequency rate plot to: {out_path}")



def plot_hourly_frequency_line(
    df: pd.DataFrame,
    out_dir: Path,
    title: str = "",
    prefix: str = "braking",
    hour_min: int = 7,
    hour_max: int = 18,
) -> None:
    """
    Stacked bar chart by hour of day.
    Each bar's stack shows **average** braking events per day by severity
    (mild/moderate/severe) in that hour window.

    Saves: <prefix>_hourly_event_severity_stacked.pdf
    """
    if df is None or df.empty:
        print("No events to plot for hourly stacked bars.")
        return

    # Ensure we have |a_min| magnitude and severity
    if "decel_mag" not in df.columns and "a_min" in df.columns:
        df = df.copy()
        df["decel_mag"] = -pd.to_numeric(df["a_min"], errors="coerce")

    if "severity" not in df.columns:
        print("Column 'severity' missing; cannot build stacked bar by severity.")
        return

    # Add local hour/date (America/New_York), drop rows without valid hour
    d = add_local_hour(df.copy())
    d = d[np.isfinite(d["hour_local"])]
    d["hour"] = d["hour_local"].astype(int)

    # Restrict to hour range
    d = d[(d["hour"] >= int(hour_min)) & (d["hour"] <= int(hour_max))].copy()
    if d.empty:
        print("No events within specified hour range for stacked bar.")
        return

    # --- NEW: number of days we are averaging over ---
    days_total = int(pd.Series(d["date_local"].dropna()).nunique())
    if days_total <= 0:
        print("Could not determine number of days for averaging; falling back to raw counts.")
        days_total = 1  # avoid division by zero; effectively raw counts

    # Keep only desired severity levels
    severity_order = ["mild", "moderate", "severe"]
    d = d[d["severity"].isin(severity_order)].copy()
    if d.empty:
        print("No events with severity in {mild, moderate, severe} to plot.")
        return

    # Build counts table: rows=hour, cols=severity
    counts = (
        d.groupby(["hour", "severity"])
         .size()
         .unstack(fill_value=0)
         .reindex(index=range(int(hour_min), int(hour_max) + 1),
                  columns=severity_order,
                  fill_value=0)
    )

    hours = counts.index.to_numpy()

    # --- NEW: convert to average per day ---
    mild = counts["mild"].to_numpy() / days_total
    moderate = counts["moderate"].to_numpy() / days_total
    severe = counts["severe"].to_numpy() / days_total
    total = mild + moderate + severe

    # Small-ish single-column friendly styling
    with mpl.rc_context({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
    }):
        fig = plt.figure(figsize=(8, 3), constrained_layout=True)

        x = np.arange(len(hours))
        bar_width = 0.65

        # Stacked bars
        plt.bar(x, mild, width=bar_width, label="Mild")
        plt.bar(x, moderate, width=bar_width, bottom=mild, label="Moderate")
        plt.bar(x, severe, width=bar_width, bottom=mild + moderate, label="Severe")

        # Dynamic headroom so text isn't up against the top
        y_max = float(total.max()) if len(total) else 0.0
        if y_max > 0:
            ylim_top = y_max * 1.15  # 15% padding
            plt.ylim(0, ylim_top)
        else:
            ylim_top = 0.0

        # Axes / labels
        print("Hourly average braking counts by severity (stacked bars):")
        plt.xlabel("Hour of day")
        plt.ylabel("Average braking events per day")  # <-- updated

        # Hour labels in AM/PM
        plt.xticks(x, [hour_to_ampm(h) for h in hours], rotation=45, ha="right")

        plt.grid(axis="y", alpha=0.3)
        plt.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            frameon=False,
        )

        # Text labels: show the average per day; use data-scaled offset
        for i in range(len(x)):
            if total[i] > 0:
                offset = 0.03 * ylim_top if ylim_top > 0 else 0.5
                plt.text(
                    x[i],
                    total[i] + offset,
                    f"{total[i]:.1f}",   # e.g., "7.5"
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        out_path = out_dir / f"{prefix}_hourly_event_severity_stacked.pdf"
        _savefig(fig, out_path)
        print(f"Saved hourly severity-stacked bar plot to: {out_path}")


def plot_severity_vs_distance_heatmap(
    df: pd.DataFrame,
    out_dir: Path,
    title: str = "",
    prefix: str = "braking",
    use_col: str = "r_start",
) -> None:
    """
    Option B: Heatmap of row-normalized proportions.
      Rows: severity (mild/moderate/severe)
      Cols: distance-to-stopbar bins using r_start (meters)

    By default uses r_start (distance at event start).
    Saves: <prefix>_severity_by_distance_heatmap.pdf
    """
    if df is None or df.empty:
        print("No events for severity-by-distance heatmap.")
        return

    if use_col not in df.columns:
        print(f"Missing {use_col} in df; cannot build severity-by-distance heatmap.")
        return

    severity_order = ["mild", "moderate", "severe"]
    dist_order = ["0–15 m", "15–30 m", "30–45 m", "45+ m"]

    def dist_bin(r: float) -> str | None:
        if not np.isfinite(r):
            return None
        r = float(r)
        if r < 0:
            return None
        if r < 15:
            return "0–15 m"
        if r < 30:
            return "15–30 m"
        if r < 45:
            return "30–45 m"
        return "45+ m"


    d = df.copy()
    d[use_col] = pd.to_numeric(d[use_col], errors="coerce")
    d["dist_bin"] = d[use_col].apply(dist_bin)

    # keep only the three severities + valid bins
    d = d[d["severity"].isin(severity_order)].dropna(subset=["dist_bin"])
    if d.empty:
        print("No valid rows after binning distance + filtering severity.")
        return

    # counts matrix
    mat = (
        d.groupby(["severity", "dist_bin"])
         .size()
         .unstack(fill_value=0)
         .reindex(index=severity_order, columns=dist_order, fill_value=0)
    )

    # row-normalized proportions
    row_sums = mat.sum(axis=1).replace(0, np.nan)
    mat_norm = mat.div(row_sums, axis=0).fillna(0.0)

    # small, one-column friendly fonts (avoid your global rcParams)
    with mpl.rc_context({
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    }):
        fig = plt.figure(figsize=(3.45, 3.05), constrained_layout=True)

        im = plt.imshow(mat_norm.values, aspect="auto", interpolation="nearest")

        plt.xticks(range(len(dist_order)), dist_order, rotation=20, ha="right")
        plt.yticks(range(len(severity_order)), severity_order)
        plt.xlabel("Distance to stopbar at event start (m)")
        plt.ylabel("Severity")
        # plt.title("Severity vs distance (row-normalized)", pad=2)
        print("Severity vs distance heatmap (row-normalized proportions)")

        cbar = plt.colorbar(im, fraction=0.05, pad=0.03)
        cbar.set_label("Proportion within severity")

        # annotate each cell with percent + (optional) raw count

        norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
        cmap = plt.get_cmap("cividis")

        for i, sev in enumerate(severity_order):
            for j, db in enumerate(dist_order):
                p = mat_norm.iloc[i, j]   # 0–1 proportion
                k = int(mat.iloc[i, j])   # raw count

                # background color for this cell
                r, g, b, _ = cmap(norm(p))
                luminance = 0.2126*r + 0.7152*g + 0.0722*b

                # --- choose text color ---
                # Make anything with reasonably high proportion use dark text,
                # regardless of what luminance thinks.
                if p >= 0.40:
                    txt_color = "black"
                else:
                    # fallback to luminance for smaller proportions
                    txt_color = "black" if luminance > 0.55 else "white"

                plt.text(
                    j,
                    i,
                    f"{100*p:.0f}%\n(n={k})",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=txt_color,
                )

        out_path = out_dir / f"{prefix}_severity_by_distance_heatmap.pdf"
        _savefig(fig, out_path)
        print(f"Saved severity-by-distance heatmap to: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--intersection-id", required=True)
    ap.add_argument("--approach-id", required=True)
    ap.add_argument("--video", default=None)
    ap.add_argument("--since", default=None, help="Only events created after this time/date.")

    ap.add_argument("--out-dir", default="outputs/braking_behavior",
                    help="Directory to save pdfs.")
    ap.add_argument("--prefix", default=None,
                    help="Filename prefix (default based on intersection).")

    ap.add_argument("--tiff", default=None,
                    help="Optional orthophoto TIFF to overlay heatmap on.")
    ap.add_argument("--heatmap-bins", type=int, default=250,
                    help="Heatmap bin resolution in pixels (default 250).")
    ap.add_argument("--heatmap-weight", choices=["count", "peak"], default="count",
                    help="Heatmap weights: count or peak (|a_min|).")

    args = ap.parse_args()

    out_dir = Path(args.out_dir)

    meta = INTERSECTION_META.get(str(args.intersection_id), {})
    default_prefix = meta.get("prefix", f"intersection_{args.intersection_id}")
    prefix = args.prefix or default_prefix

    CH_HOST = os.getenv("CH_HOST", "example.com")
    CH_PORT = int(os.getenv("CH_PORT", "8123"))
    CH_USER = os.getenv("CH_USER", "default")
    CH_PASSWORD = os.getenv("CH_PASSWORD", "")
    CH_DB = os.getenv("CH_DB", "trajectories")

    ch = ClickHouseHTTP(
        host=CH_HOST,
        port=CH_PORT,
        user=CH_USER,
        password=CH_PASSWORD,
        database=CH_DB,
    )

    df = fetch_events_df(
        ch,
        intersection_id=args.intersection_id,
        approach_id=args.approach_id,
        video=args.video,
        since=args.since,
    )

    print(quantify_periods(df, restrict_hours_7_to_18_for_corr=True))

    meta = INTERSECTION_META.get(str(args.intersection_id), {})
    inter_name = meta.get("tex_name", f"Intersection {args.intersection_id}")
    base_title = f"at {inter_name}"
    title = base_title + (f" (video={args.video})" if args.video else "")

    plot_dashboard(df, out_dir=out_dir, title=title, prefix=prefix)

    plot_frequency_rates(df, out_dir=out_dir, title=title, prefix=prefix)

    plot_hourly_frequency_line(df, out_dir=out_dir, title=title, prefix=prefix, hour_min=7, hour_max=18)

    print_hourly_stacked_latex(df, intersection_id=args.intersection_id, prefix=prefix)

    plot_severity_vs_distance_heatmap(df, out_dir=out_dir, title=title, prefix=prefix, use_col="r_start")



    # NOTE:
    # We deliberately build the overlay in ORTHO PIXEL space, not from x/y meters.
    # The pipeline camera → homography H → ClickHouse (map_px_x/map_px_y) is already
    # calibrated and produces correctly aligned (col_px,row_px) points on the TIFF.
    # Earlier misalignment came from re-projecting x_evt/y_evt (meters) back through
    # the GeoTIFF affine; instead we now:
    #   1) JOIN braking_events → raw to get per-event map_px_x/map_px_y,
    #   2) aggregate to col_px,row_px,
    #   3) render the heatmap directly in that pixel grid on top of the TIFF image.

    if args.tiff:
        tiff_path = Path(args.tiff)

        # 1) Get per-event pixel positions from raw table
        dfp = fetch_event_positions_df(
            ch,
            intersection_id=args.intersection_id,
            approach_id=args.approach_id,
            video=args.video,
            since=args.since,
        )

        if dfp.empty:
            print("No positioned events for heatmap.")
        else:
            heat_out = out_dir / f"{prefix}_heatmap_overlay_{args.heatmap_weight}.pdf"
            plot_heatmap_on_tiff(
                dfp=dfp,
                tiff_path=tiff_path,
                out_path=heat_out,
                bins=args.heatmap_bins,
                weight_mode=args.heatmap_weight,
            )



    if not df.empty:
        print("\n--- Summary ---")
        print("Events:", len(df))
        print("Median |a_min| (m/s²):", float(np.nanmedian(df["decel_mag"])))
        print("90th pct |a_min|:", float(np.nanpercentile(df["decel_mag"], 90)))
        print("Median Δv (m/s):", float(np.nanmedian(df["dv"])))
        print("90th pct Δv:", float(np.nanpercentile(df["dv"], 90)))


if __name__ == "__main__":
    main()
