import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

def validate_energy_data(df, energy_type):
    """
    Validate the uploaded energy data file with comprehensive data quality checks
    Returns (is_valid, message, cleaned_df, detected_interval, validation_stats)
    """
    errors = []
    warnings = []
    validation_stats = {}
    
    # Check if required columns exist
    required_columns = ['timestamp', 'energy_consumption']  # Adjust as needed
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
        return False, "; ".join(errors), None, None, {}
    
    # Store original data info
    original_row_count = len(df)
    validation_stats['original_rows'] = original_row_count
    
    # Try to parse timestamp column with multiple formats
    try:
        # First try automatic parsing with UTC inference
        df['timestamp'] = pd.to_datetime(df['timestamp'], infer_datetime_format=True, utc=True)
        # Convert to local timezone if UTC was applied automatically
        if df['timestamp'].dt.tz is not None:
            df['timestamp'] = df['timestamp'].dt.tz_convert(None)  # Remove timezone info for consistency
    except:
        # Try common European formats
        common_formats = [
            '%d-%m-%Y %H:%M:%S',  # 01-12-2024 14:30:00
            '%d/%m/%Y %H:%M:%S',  # 01/12/2024 14:30:00
            '%d-%m-%Y %H:%M',     # 01-12-2024 14:30
            '%d/%m/%Y %H:%M',     # 01/12/2024 14:30
            '%Y-%m-%d %H:%M:%S',  # 2024-12-01 14:30:00
            '%Y/%m/%d %H:%M:%S',  # 2024/12/01 14:30:00
            '%Y-%m-%d %H:%M',     # 2024-12-01 14:30
            '%Y/%m/%d %H:%M',     # 2024/12/01 14:30
            '%d.%m.%Y %H:%M:%S',  # 01.12.2024 14:30:00
            '%d.%m.%Y %H:%M',     # 01.12.2024 14:30
        ]
        
        parsed = False
        for fmt in common_formats:
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'], format=fmt)
                parsed = True
                break
            except:
                continue
        
        if not parsed:
            # Last attempt with dayfirst=True for European dates
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True)
                parsed = True
            except:
                pass
        
        if not parsed:
            errors.append("Invalid timestamp format. Please use formats like 'YYYY-MM-DD HH:MM:SS' or 'DD-MM-YYYY HH:MM:SS'")
            return False, "; ".join(errors), None, None, {}
    
    # Handle timezone information
    original_tz = None
    if df['timestamp'].dt.tz is not None:
        original_tz = str(df['timestamp'].dt.tz)
        warnings.append(f"Timezone detected: {original_tz}. Converting to local time for analysis.")
        # Convert to naive datetime for consistent processing
        df['timestamp'] = df['timestamp'].dt.tz_localize(None)
    
    # Sort by timestamp for analysis
    df = df.sort_values('timestamp')
    
    # Check if we have any data after timestamp parsing
    if len(df) == 0:
        errors.append("No valid timestamps found in the data")
        return False, "; ".join(errors), None, None, {'original_rows': original_row_count}
    
    # DATA RANGE ANALYSIS
    first_date = df['timestamp'].min()
    last_date = df['timestamp'].max()
    total_duration = last_date - first_date
    validation_stats['date_range'] = {
        'first_date': first_date,
        'last_date': last_date,
        'duration_days': total_duration.days,
        'duration_hours': total_duration.total_seconds() / 3600
    }
    
    # Detect data interval (15-min vs hourly)
    time_diffs = df['timestamp'].diff().dropna()
    
    # Check if we have any time differences to analyze
    if len(time_diffs) == 0:
        errors.append("Need at least 2 data points to determine interval")
        return False, "; ".join(errors), None, None, validation_stats
    
    # Calculate most common interval
    intervals_minutes = time_diffs.dt.total_seconds() / 60
    interval_counts = intervals_minutes.value_counts()
    
    if len(interval_counts) == 0:
        errors.append("Cannot determine data interval from timestamps")
        return False, "; ".join(errors), None, None, validation_stats
    
    most_common_interval = interval_counts.index[0]
    
    # Determine if it's 15-min or hourly data
    if abs(most_common_interval - 15) <= 2:  # 15-minute data (with 2-min tolerance)
        expected_interval = timedelta(minutes=15)
        detected_interval = "15-minute"
        tolerance = timedelta(minutes=2)
        expected_daily_points = 96
    elif abs(most_common_interval - 60) <= 5:  # Hourly data (with 5-min tolerance)
        expected_interval = timedelta(minutes=60)
        detected_interval = "hourly"
        tolerance = timedelta(minutes=5)
        expected_daily_points = 24
    else:
        # Check for other common intervals
        if abs(most_common_interval - 30) <= 2:  # 30-minute data
            expected_interval = timedelta(minutes=30)
            detected_interval = "30-minute"
            tolerance = timedelta(minutes=2)
            expected_daily_points = 48
        elif abs(most_common_interval - 5) <= 1:  # 5-minute data
            expected_interval = timedelta(minutes=5)
            detected_interval = "5-minute"
            tolerance = timedelta(minutes=1)
            expected_daily_points = 288
        else:
            warnings.append(f"Unusual data interval detected: {most_common_interval:.1f} minutes. Proceeding with analysis.")
            expected_interval = timedelta(minutes=most_common_interval)
            detected_interval = f"{most_common_interval:.1f}-minute"
            tolerance = timedelta(minutes=max(1, most_common_interval * 0.1))
            expected_daily_points = int(24 * 60 / most_common_interval)
    
    # DATA GAP ANALYSIS
    if total_duration.total_seconds() > 0:
        expected_total_points = int((total_duration.total_seconds() / 60) / most_common_interval) + 1
        actual_points = len(df)
        missing_points = max(0, expected_total_points - actual_points)
        completeness_pct = (actual_points / expected_total_points * 100) if expected_total_points > 0 else 100
    else:
        expected_total_points = len(df)
        actual_points = len(df)
        missing_points = 0
        completeness_pct = 100
    
    validation_stats['data_gaps'] = {
        'expected_points': expected_total_points,
        'actual_points': actual_points,
        'missing_points': missing_points,
        'completeness_pct': completeness_pct
    }
    
    # Find actual gaps (periods longer than expected interval + tolerance)
    large_gaps = time_diffs[time_diffs > expected_interval + tolerance]
    gap_details = []
    if len(large_gaps) > 0 and len(df) > 1:
        # Get the timestamps where gaps occur
        for gap_timestamp in large_gaps.index:
            gap_duration = large_gaps[gap_timestamp]
            # Find the previous timestamp to calculate gap start
            try:
                gap_start_idx = df[df['timestamp'] == gap_timestamp].index[0] - 1
                if gap_start_idx >= 0 and gap_start_idx < len(df):
                    gap_start_time = df.iloc[gap_start_idx]['timestamp']
                    gap_details.append({
                        'start_time': gap_start_time,
                        'duration_hours': gap_duration.total_seconds() / 3600
                    })
            except (IndexError, KeyError):
                # Skip this gap if we can't determine the start time
                continue
    
    validation_stats['gap_details'] = gap_details
    
    # Check interval consistency
    if len(time_diffs) > 0:
        irregular_intervals = time_diffs[(time_diffs < expected_interval - tolerance) | 
                                       (time_diffs > expected_interval + tolerance)]
        
        if len(irregular_intervals) > len(time_diffs) * 0.1:  # More than 10% irregular
            warnings.append(f"Some irregular intervals detected in {detected_interval} data ({len(irregular_intervals)} out of {len(time_diffs)} intervals)")
    
    # ENERGY VALUES ANALYSIS
    try:
        # Store original values for comparison
        original_energy_values = df['energy_consumption'].copy()
        
        # Handle European decimal separator (comma instead of dot)
        # First, check if values are strings and contain commas
        if df['energy_consumption'].dtype == 'object':
            # Check if we have comma decimal separators
            sample_values = df['energy_consumption'].dropna().head(100)
            comma_count = sum(str(val).count(',') for val in sample_values)
            dot_count = sum(str(val).count('.') for val in sample_values)
            
            if comma_count > dot_count and comma_count > 0:
                # European format: replace comma with dot for decimal separator
                df['energy_consumption'] = df['energy_consumption'].astype(str).str.replace(',', '.', regex=False)
                warnings.append("Detected European decimal format (comma separator). Converting to standard format.")
        
        # Convert to numeric, keeping track of conversion issues
        df['energy_consumption'] = pd.to_numeric(df['energy_consumption'], errors='coerce')
        
        # Count conversion failures (NaNs created by pd.to_numeric)
        conversion_nans = df['energy_consumption'].isna().sum() - original_energy_values.isna().sum()
        if conversion_nans > 0:
            warnings.append(f"{conversion_nans} non-numeric {energy_type.lower()} values converted to NaN")
            
    except Exception as e:
        errors.append(f"{energy_type} consumption values must be numeric: {str(e)}")
    
    # NaN ANALYSIS
    nan_count = df['energy_consumption'].isna().sum()
    validation_stats['data_quality'] = {
        'total_nans': nan_count,
        'nan_percentage': (nan_count / len(df) * 100) if len(df) > 0 else 0,
        'conversion_nans': conversion_nans if 'conversion_nans' in locals() else 0
    }
    
    # ZERO VALUES ANALYSIS
    non_nan_mask = ~df['energy_consumption'].isna()
    zero_count = (df.loc[non_nan_mask, 'energy_consumption'] == 0).sum()
    validation_stats['zero_analysis'] = {
        'zero_count': zero_count,
        'zero_percentage': (zero_count / non_nan_mask.sum() * 100) if non_nan_mask.sum() > 0 else 0
    }
    
    # OUTLIER ANALYSIS (using IQR method)
    clean_values = df['energy_consumption'].dropna()
    if len(clean_values) > 0:
        q1 = clean_values.quantile(0.25)
        q3 = clean_values.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers_mask = (clean_values < lower_bound) | (clean_values > upper_bound)
        outlier_count = outliers_mask.sum()
        
        # Extreme outliers (3 * IQR)
        extreme_lower = q1 - 3 * iqr
        extreme_upper = q3 + 3 * iqr
        extreme_outliers_mask = (clean_values < extreme_lower) | (clean_values > extreme_upper)
        extreme_outlier_count = extreme_outliers_mask.sum()
        
        validation_stats['outliers'] = {
            'mild_outliers': outlier_count - extreme_outlier_count,
            'extreme_outliers': extreme_outlier_count,
            'total_outliers': outlier_count,
            'outlier_percentage': (outlier_count / len(clean_values) * 100) if len(clean_values) > 0 else 0,
            'q1': q1,
            'q3': q3,
            'iqr': iqr,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'min_value': clean_values.min(),
            'max_value': clean_values.max(),
            'median': clean_values.median(),
            'mean': clean_values.mean(),
            'std': clean_values.std()
        }
        
        if extreme_outlier_count > 0:
            warnings.append(f"{extreme_outlier_count} extreme outliers detected (>3×IQR from median)")
        elif outlier_count > len(clean_values) * 0.05:  # More than 5% outliers
            warnings.append(f"{outlier_count} potential outliers detected ({outlier_count/len(clean_values)*100:.1f}% of data)")
    else:
        validation_stats['outliers'] = {'total_outliers': 0, 'outlier_percentage': 0}
    
    # COMPLETENESS ANALYSIS
    if validation_stats['data_gaps']['completeness_pct'] < 95:
        warnings.append(f"Data completeness is {validation_stats['data_gaps']['completeness_pct']:.1f}% - significant gaps detected")
    elif validation_stats['data_gaps']['completeness_pct'] < 99:
        warnings.append(f"Data completeness is {validation_stats['data_gaps']['completeness_pct']:.1f}% - minor gaps detected")
    
    # Check for reasonable data range
    if len(clean_values) > 0:
        if clean_values.max() < 0:
            errors.append(f"All {energy_type.lower()} consumption values are negative")
        elif clean_values.min() < 0:
            negative_count = (clean_values < 0).sum()
            warnings.append(f"{negative_count} negative {energy_type.lower()} consumption values detected")
    
    # Remove rows with missing values for final dataset
    clean_df = df.dropna()
    validation_stats['final_clean_rows'] = len(clean_df)
    
    if len(clean_df) == 0:
        errors.append("No valid data rows remaining after cleaning")
        return False, "; ".join(errors), None, None, validation_stats
    
    # Prepare success message
    success_msg = f"Data validation successful. Detected {detected_interval} intervals."
    if original_tz:
        success_msg += f" Original timezone: {original_tz}."
    if warnings:
        success_msg += f" Warnings: {'; '.join(warnings)}"
    
    if errors:
        return False, "; ".join(errors), clean_df, detected_interval, validation_stats
    else:
        return True, success_msg, clean_df, detected_interval, validation_stats

def get_season(date):
    """
    Determine season based on date (Northern Hemisphere)
    """
    month = date.month
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Fall'

def analyze_energy_data(df, detected_interval, energy_type, energy_unit):
    """
    Perform energy consumption analysis
    Returns dictionary with analysis results
    """
    # Basic statistics
    total_consumption = df['energy_consumption'].sum()
    avg_consumption = df['energy_consumption'].mean()
    max_consumption = df['energy_consumption'].max()
    min_consumption = df['energy_consumption'].min()
    
    # Time-based analysis
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.day_name()
    df['date'] = df['timestamp'].dt.date
    
    # Seasonal analysis
    df['season'] = df['timestamp'].dt.date.apply(get_season)
    seasonal_hourly_avg = df.groupby(['season', 'hour'])['energy_consumption'].mean().unstack(level=0)
    
    # Daily patterns
    hourly_avg = df.groupby('hour')['energy_consumption'].mean()
    daily_total = df.groupby('date')['energy_consumption'].sum()
    weekly_avg = df.groupby('day_of_week')['energy_consumption'].mean()
    
    # Peak analysis
    peak_hour = hourly_avg.idxmax()
    peak_consumption = hourly_avg.max()
    
    # Calculate proper units based on interval
    if detected_interval == "15-minute":
        interval_text = "15-min interval"
        daily_intervals = 96  # 24 hours * 4 intervals per hour
    elif detected_interval == "hourly":
        interval_text = "hourly interval"
        daily_intervals = 24  # 24 hours
    elif detected_interval == "30-minute":
        interval_text = "30-min interval"
        daily_intervals = 48  # 24 hours * 2 intervals per hour
    elif detected_interval == "5-minute":
        interval_text = "5-min interval"
        daily_intervals = 288  # 24 hours * 12 intervals per hour
    else:
        interval_text = f"{detected_interval} interval"
        # Estimate daily intervals
        interval_minutes = float(detected_interval.split('-')[0])
        daily_intervals = int(24 * 60 / interval_minutes)
    
    return {
        'total_consumption': total_consumption,
        'avg_consumption': avg_consumption,
        'max_consumption': max_consumption,
        'min_consumption': min_consumption,
        'peak_hour': peak_hour,
        'peak_consumption': peak_consumption,
        'hourly_avg': hourly_avg,
        'daily_total': daily_total,
        'weekly_avg': weekly_avg,
        'seasonal_hourly_avg': seasonal_hourly_avg,  # New addition
        'data_period': f"{df['timestamp'].min().strftime('%Y-%m-%d')} to {df['timestamp'].max().strftime('%Y-%m-%d')}",
        'detected_interval': detected_interval,
        'interval_text': interval_text,
        'daily_intervals': daily_intervals,
        'energy_type': energy_type,
        'energy_unit': energy_unit
    }

def create_visualizations(analysis_results):
    """
    Create matplotlib visualizations optimized for 2x2 grid layout
    Returns list of figure objects
    """
    figures = []
    energy_type = analysis_results['energy_type']
    energy_unit = analysis_results['energy_unit']
    
    # Clear any existing plots
    plt.clf()
    plt.close('all')
    
    # Smaller figure sizes for 2x2 layout (reduced from original sizes)
    
    # Daily totals over time
    fig1, ax1 = plt.subplots(figsize=(8, 5))  # Reduced from (12, 6)
    daily_total = analysis_results['daily_total']
    ax1.plot(daily_total.index, daily_total.values, marker='o', markersize=3)
    ax1.set_title(f'Total daily {energy_type} consumption', fontsize=12)
    ax1.set_xlabel('Date', fontsize=10)
    ax1.set_ylabel(f'Total {energy_type} consumption ({energy_unit})', fontsize=10)
    ax1.tick_params(axis='both', which='major', labelsize=9)
    plt.xticks(rotation=45)
    plt.tight_layout()
    figures.append(fig1)
    
    # Daily consumption pattern
    fig2, ax2 = plt.subplots(figsize=(8, 5))  # Reduced from (10, 6)
    analysis_results['hourly_avg'].plot(kind='line', ax=ax2, linewidth=2)
    ax2.set_title(f'Average {energy_type} consumption by hour of day', fontsize=12)
    ax2.set_xlabel('Hour of day', fontsize=10)
    ax2.set_ylabel(f'{energy_type} Consumption ({energy_unit})', fontsize=10)
    ax2.tick_params(axis='both', which='major', labelsize=9)
    plt.xticks(rotation=0)
    plt.tight_layout()
    figures.append(fig2)
    
    # Weekly pattern
    fig3, ax3 = plt.subplots(figsize=(8, 5))  # Reduced from (10, 6)
    analysis_results['weekly_avg'].plot(kind='line', ax=ax3, color='orange', linewidth=2, marker='o')
    ax3.set_title(f'Average {energy_type} Consumption by day of week', fontsize=12)
    ax3.set_xlabel('Day of week', fontsize=10)
    ax3.set_ylabel(f'{energy_type} Consumption ({energy_unit})', fontsize=10)
    ax3.tick_params(axis='both', which='major', labelsize=9)
    plt.xticks(rotation=45)
    plt.tight_layout()
    figures.append(fig3)
    
    # Seasonal hourly consumption pattern
    fig4, ax4 = plt.subplots(figsize=(8, 5))  # Reduced from (12, 8)
    seasonal_data = analysis_results['seasonal_hourly_avg']
    
    # Define colors for each season
    season_colors = {
        'Spring': '#90EE90',  # Light green
        'Summer': '#FFD700',  # Gold
        'Fall': '#FF8C00',    # Dark orange
        'Winter': '#87CEEB'   # Sky blue
    }
    
    # Plot each season that exists in the data
    for season in seasonal_data.columns:
        if season in season_colors:
            ax4.plot(seasonal_data.index, seasonal_data[season], 
                    label=season, linewidth=2, marker='o', markersize=3,
                    color=season_colors[season])
    
    ax4.set_title(f'Average {energy_type} Consumption by Hour - Seasonal', fontsize=12)  # Shortened title
    ax4.set_xlabel('Hour of Day', fontsize=10)
    ax4.set_ylabel(f'{energy_type} Consumption ({energy_unit})', fontsize=10)
    ax4.tick_params(axis='both', which='major', labelsize=9)
    ax4.legend(title='Season', fontsize=9, title_fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(range(0, 24, 4))  # Show fewer x-axis labels for cleaner look
    plt.tight_layout()
    figures.append(fig4)
    
    return figures

def generate_pdf_report(analysis_results, figures, validation_stats):
    """
    Generate PDF report using ReportLab with SECURE in-memory image handling
    Returns bytes of PDF file - NO temporary files created on disk
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    energy_type = analysis_results['energy_type']
    energy_unit = analysis_results['energy_unit']
    
    # Title
    title = Paragraph(f"{energy_type} Energy savings potential report", styles['Title'])
    story.append(title)
    story.append(Spacer(1, 12))
    
    # Summary section
    summary_title = Paragraph("Summary of inputs and statistics", styles['Heading1'])
    story.append(summary_title)
    
    summary_text = f"""
    <para>
    <b>Energy Type:</b> {energy_type}<br/>
    <b>Analysis Period:</b> {analysis_results['data_period']}<br/>
    <b>Data Interval:</b> {analysis_results['detected_interval']}<br/>
    <b>Total {energy_type} Consumption:</b> {analysis_results['total_consumption']:.2f} {energy_unit}<br/>
    <b>Average Consumption:</b> {analysis_results['avg_consumption']:.2f} {energy_unit} per {analysis_results['interval_text']}<br/>
    <b>Peak Hour:</b> {analysis_results['peak_hour']}:00 ({analysis_results['peak_consumption']:.2f} {energy_unit} average)<br/>
    <b>Maximum {analysis_results['interval_text']} Consumption:</b> {analysis_results['max_consumption']:.2f} {energy_unit}<br/>
    <b>Minimum {analysis_results['interval_text']} Consumption:</b> {analysis_results['min_consumption']:.2f} {energy_unit}
    </para>
    """
    
    summary_para = Paragraph(summary_text, styles['Normal'])
    story.append(summary_para)
    story.append(Spacer(1, 20))
    
    # Data Quality Section
    quality_title = Paragraph("Data quality assessment", styles['Heading1'])
    story.append(quality_title)
    
    # Build data quality text
    quality_items = []
    
    # Date range and completeness
    if validation_stats.get('date_range'):
        dr = validation_stats['date_range']
        quality_items.append(f"<b>Data Coverage:</b> {dr['duration_days']} days ({dr['duration_hours']:.1f} hours)")
    
    if validation_stats.get('data_gaps'):
        dg = validation_stats['data_gaps'] 
        completeness_status = "Excellent" if dg['completeness_pct'] >= 95 else "Good" if dg['completeness_pct'] >= 90 else "Fair" if dg['completeness_pct'] >= 80 else "Poor"
        quality_items.append(f"<b>Data Completeness:</b> {dg['completeness_pct']:.1f}% ({completeness_status})")
        quality_items.append(f"<b>Expected Data Points:</b> {dg['expected_points']:,}")
        quality_items.append(f"<b>Actual Data Points:</b> {dg['actual_points']:,}")
        if dg['missing_points'] > 0:
            quality_items.append(f"<b>Missing Data Points:</b> {dg['missing_points']:,}")
    
    # Data quality issues
    if validation_stats.get('data_quality'):
        dq = validation_stats['data_quality']
        if dq['total_nans'] > 0:
            quality_items.append(f"<b>Invalid Values:</b> {dq['total_nans']:,} ({dq['nan_percentage']:.1f}%)")
    
    if validation_stats.get('zero_analysis'):
        za = validation_stats['zero_analysis']
        if za['zero_count'] > 0:
            quality_items.append(f"<b>Zero Values:</b> {za['zero_count']:,} ({za['zero_percentage']:.1f}%)")
    
    # Outliers
    if validation_stats.get('outliers'):
        ol = validation_stats['outliers']
        if ol['total_outliers'] > 0:
            outlier_status = "Low" if ol['outlier_percentage'] < 2 else "Moderate" if ol['outlier_percentage'] < 5 else "High"
            quality_items.append(f"<b>Outliers:</b> {ol['total_outliers']:,} ({ol['outlier_percentage']:.1f}%) - {outlier_status} impact")
            if ol.get('extreme_outliers', 0) > 0:
                quality_items.append(f"<b>Extreme Outliers:</b> {ol['extreme_outliers']:,} (may indicate data issues)")
    
    # Data gaps
    if validation_stats.get('gap_details') and len(validation_stats['gap_details']) > 0:
        gap_count = len(validation_stats['gap_details'])
        quality_items.append(f"<b>Significant Data Gaps:</b> {gap_count} gaps detected")
        
        # Show largest gaps
        gaps_df = pd.DataFrame(validation_stats['gap_details'])
        largest_gap = gaps_df['duration_hours'].max()
        if largest_gap >= 24:
            quality_items.append(f"<b>Largest Gap:</b> {largest_gap/24:.1f} days")
        else:
            quality_items.append(f"<b>Largest Gap:</b> {largest_gap:.1f} hours")
    
    # Statistical summary
    if validation_stats.get('outliers'):
        ol = validation_stats['outliers']
        if 'median' in ol:
            quality_items.append(f"<b>Statistical Range:</b> {ol['min_value']:.2f} - {ol['max_value']:.2f} {energy_unit}")
            quality_items.append(f"<b>Median Consumption:</b> {ol['median']:.2f} {energy_unit}")
            quality_items.append(f"<b>Standard Deviation:</b> {ol['std']:.2f} {energy_unit}")
    
    # Create quality text
    quality_text = "<para>" + "<br/>".join(quality_items) + "</para>"
    quality_para = Paragraph(quality_text, styles['Normal'])
    story.append(quality_para)
    story.append(Spacer(1, 20))
    
    # Data Quality Recommendations
    recommendations = []
    if validation_stats.get('data_gaps', {}).get('completeness_pct', 100) < 95:
        recommendations.append("• Consider investigating data collection gaps for more complete analysis")
    if validation_stats.get('outliers', {}).get('extreme_outliers', 0) > 0:
        recommendations.append("• Review extreme outliers to identify potential measurement errors or unusual events")
    if validation_stats.get('zero_analysis', {}).get('zero_percentage', 0) > 10:
        recommendations.append("• High percentage of zero values may indicate measurement issues or system downtime")
    if len(validation_stats.get('gap_details', [])) > 5:
        recommendations.append("• Multiple data gaps detected - consider improving data collection reliability")
    
    if recommendations:
        rec_title = Paragraph("Data quality recommendations", styles['Heading2'])
        story.append(rec_title)
        rec_text = "<para>" + "<br/>".join(recommendations) + "</para>"
        rec_para = Paragraph(rec_text, styles['Normal'])
        story.append(rec_para)
        story.append(Spacer(1, 20))
    
    # SECURE CHART HANDLING - NO TEMP FILES, PURE MEMORY APPROACH
    for i, fig in enumerate(figures):
        try:
            # Create PNG image in memory buffer - NEVER touches disk
            img_buffer = io.BytesIO()
            fig.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)  # Reset buffer position to beginning
            
            # Create ReportLab Image directly from memory buffer
            img = Image(img_buffer, width=500, height=300)
            story.append(img)
            story.append(Spacer(1, 12))
            
            # img_buffer automatically garbage collected when out of scope
            # No cleanup needed - no files were created!
            
        except Exception as e:
            # If chart generation fails, add placeholder text instead of crashing
            error_para = Paragraph(f"[Chart {i+1} could not be generated: {type(e).__name__}]", styles['Normal'])
            story.append(error_para)
            story.append(Spacer(1, 12))
            # Log error for debugging (optional)
            print(f"Chart {i+1} generation failed: {e}")
    
    # Build PDF - all processing done in memory
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

# Streamlit App
st.set_page_config(page_title="Energy consumption report and savings potential generator", layout="wide")

st.title("🔋 Energy consumption report and savings potential generator")
st.markdown("Upload your energy (gas/electricity/heat) consumption data to generate an automated analysis report.")

# Energy type selection
energy_type = st.selectbox(
    "Select Energy Type",
    ["Electricity", "Gas", "Heat"],
    index=0,
    help="Choose the type of energy data you're uploading"
)

# Set units based on energy type
energy_units = {
    "Electricity": "kWh",
    "Gas": "m³", 
    "Heat": "GJ"
}

current_unit = energy_units[energy_type]
st.info(f"Selected: **{energy_type}** (measured in **{current_unit}**)")

# File upload
uploaded_file = st.file_uploader(
    "Choose a CSV file with energy consumption data",
    type=['csv'],
    help="File should contain 'timestamp' and 'energy_consumption' columns with 15-minute intervals"
)

if uploaded_file is not None:
    try:
        # Try to detect the delimiter automatically
        sample = uploaded_file.read(1024).decode('utf-8')
        uploaded_file.seek(0)  # Reset file pointer
        
        # Check for common delimiters
        if ';' in sample and sample.count(';') > sample.count(','):
            delimiter = ';'
        elif ',' in sample:
            delimiter = ','
        elif '\t' in sample:
            delimiter = '\t'
        else:
            delimiter = ','  # Default fallback
        
        # Read the uploaded file with detected delimiter
        df = pd.read_csv(uploaded_file, sep=delimiter)
        
        st.info(f"Detected delimiter: '{delimiter}'")
        
        st.subheader("📋 Data validation")
        
        # Validate the data
        is_valid, message, clean_df, detected_interval, validation_stats = validate_energy_data(df, energy_type)
        
        if is_valid:
            st.success(f"✅ {message}")
            st.info(f"Dataset contains {len(clean_df)} valid data points with {detected_interval} intervals")
            
            # Display detailed validation statistics
            with st.expander("📊 Data quality report", expanded=True):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**📅 Date range**")
                    if validation_stats.get('date_range'):
                        dr = validation_stats['date_range']
                        st.write(f"**From:** {dr['first_date'].strftime('%Y-%m-%d %H:%M')}")
                        st.write(f"**To:** {dr['last_date'].strftime('%Y-%m-%d %H:%M')}")
                        st.write(f"**Duration:** {dr['duration_days']} days ({dr['duration_hours']:.1f} hours)")
                
                with col2:
                    st.markdown("**🔍 Data completeness**")
                    if validation_stats.get('data_gaps'):
                        dg = validation_stats['data_gaps']
                        st.write(f"**Expected points:** {dg['expected_points']:,}")
                        st.write(f"**Actual points:** {dg['actual_points']:,}")
                        st.write(f"**Missing points:** {dg['missing_points']:,}")
                        completeness_color = "🟢" if dg['completeness_pct'] >= 95 else "🟡" if dg['completeness_pct'] >= 90 else "🔴"
                        st.write(f"**Completeness:** {completeness_color} {dg['completeness_pct']:.1f}%")
                
                with col3:
                    st.markdown("**⚠️ Data issues**")
                    if validation_stats.get('data_quality'):
                        dq = validation_stats['data_quality']
                        st.write(f"**NaN values:** {dq['total_nans']:,} ({dq['nan_percentage']:.1f}%)")
                    
                    if validation_stats.get('zero_analysis'):
                        za = validation_stats['zero_analysis']
                        st.write(f"**Zero values:** {za['zero_count']:,} ({za['zero_percentage']:.1f}%)")
                    
                    if validation_stats.get('outliers'):
                        ol = validation_stats['outliers']
                        outlier_color = "🟢" if ol['outlier_percentage'] < 2 else "🟡" if ol['outlier_percentage'] < 5 else "🔴"
                        st.write(f"**Outliers:** {outlier_color} {ol['total_outliers']:,} ({ol['outlier_percentage']:.1f}%)")
                
                # Data gaps details
                if validation_stats.get('gap_details') and len(validation_stats['gap_details']) > 0:
                    st.markdown("**📊 Significant Data Gaps**")
                    gaps_df = pd.DataFrame(validation_stats['gap_details'])
                    gaps_df['duration_formatted'] = gaps_df['duration_hours'].apply(
                        lambda x: f"{x:.1f} hours" if x < 24 else f"{x/24:.1f} days"
                    )
                    st.dataframe(
                        gaps_df[['start_time', 'duration_formatted']].rename(columns={
                            'start_time': 'Gap Start',
                            'duration_formatted': 'Duration'
                        }),
                        hide_index=True
                    )
                
                # Outlier details
                if validation_stats.get('outliers') and validation_stats['outliers']['total_outliers'] > 0:
                    st.markdown("**📈 Statistical summary**")
                    ol = validation_stats['outliers']
                    stats_col1 = st.columns(1)
                    
                    with stats_col1:
                        st.write(f"**Min:** {ol['min_value']:.2f} {current_unit}")
                        st.write(f"**Mean:** {ol['mean']:.2f} {current_unit}")
                        st.write(f"**Max:** {ol['max_value']:.2f} {current_unit}")
                        st.write(f"**Median:** {ol['median']:.2f} {current_unit}")
                    
                    if ol['extreme_outliers'] > 0:
                        st.warning(f"⚠️ {ol['extreme_outliers']} extreme outliers detected (>3×IQR). These may indicate data errors or unusual consumption events.")
            
            # Show data preview
            with st.expander("Preview uploaded data"):
                st.dataframe(clean_df.head(10))
            
            # Perform analysis
            st.subheader("📊 Analysis results")
            
            with st.spinner(f"Analyzing {energy_type.lower()} consumption data..."):
                analysis_results = analyze_energy_data(clean_df, detected_interval, energy_type, current_unit)
                figures = create_visualizations(analysis_results)
            
            # Display key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(f"Total {energy_type}", f"{analysis_results['total_consumption']:.2f} {current_unit}")
            
            with col2:
                st.metric(f"Average per {analysis_results['interval_text']}", f"{analysis_results['avg_consumption']:.2f} {current_unit}")
            
            with col3:
                st.metric("Peak Hour", f"{analysis_results['peak_hour']}:00")
            
            with col4:
                st.metric(f"Peak {energy_type}", f"{analysis_results['peak_consumption']:.2f} {current_unit}")
            
            # Display visualizations in 2x2 grid
            st.subheader("📈 Visualizations")
            
            # First row - two columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Daily Total Consumption")
                if len(figures) > 0:
                    st.pyplot(figures[0])
                    plt.close(figures[0])
            
            with col2:
                st.subheader("Hourly Consumption Pattern")
                if len(figures) > 1:
                    st.pyplot(figures[1])
                    plt.close(figures[1])
            
            # Second row - two columns
            col3, col4 = st.columns(2)
            
            with col3:
                st.subheader("Weekly Consumption Pattern")
                if len(figures) > 2:
                    st.pyplot(figures[2])
                    plt.close(figures[2])
            
            with col4:
                st.subheader("Seasonal Consumption Pattern")
                if len(figures) > 3:
                    st.pyplot(figures[3])
                    plt.close(figures[3])
            
            # Generate and offer PDF download
            st.subheader("📄 Generate Report")
            
            if st.button("📥 Generate PDF Report", type="primary"):
                with st.spinner("Generating PDF report..."):
                    pdf_bytes = generate_pdf_report(analysis_results, figures, validation_stats)
                
                st.download_button(
                    label="📥 Download PDF report",  
                    data=pdf_bytes,
                    file_name=f"{energy_type.lower()}_consumption_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
                
                st.success("✅ PDF report generated successfully!")
        
        else:
            st.error(f"❌ Data validation failed: {message}")
            st.info("Please check your data format and try again.")
            
            # Show data preview for debugging
            with st.expander("Show uploaded data for debugging"):
                st.dataframe(df.head())
    
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.info("Please ensure your file is a valid CSV with the correct format.")

else:
    st.info("👆 Please upload a CSV file to get started.")
    
    # Show expected format
    with st.expander("Expected Data Format"):
        st.markdown(f"""
        Your CSV file should have the following structure for **{energy_type}** data:
        
        | timestamp | energy_consumption |
        |-----------|-------------------|
        | 2024-01-01 00:00:00 | 2.5 |
        | 2024-01-01 00:15:00 | 2.3 |
        | 2024-01-01 00:30:00 | 2.7 |
        | ... | ... |
        
        **Requirements:**
        - `timestamp` column with datetime values (supports timezones)
        - `energy_consumption` column with numeric values (**{current_unit}**)
        - CSV format with headers
        - Supports 15-minute, 30-minute, hourly, and other regular intervals
        - Handles various date formats (DD-MM-YYYY, YYYY-MM-DD, etc.)
        - Automatically detects delimiter (; or , or tab)
        - **Supports European decimal format** (comma as decimal separator: 2,5 instead of 2.5)
        """)

# Footer
st.markdown("---")
st.markdown("Built with Streamlit • Upload your energy data to generate automated consumption reports • Supports Electricity (kWh), Gas (m³), and Heat (GJ)")