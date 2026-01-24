"""
Visualization utilities for TR Data Challenge.
Creates professional, publication-ready plots and charts.
"""

from typing import Optional, Union
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import numpy as np

from .data_analyzer import DatasetStatistics


# ═══════════════════════════════════════════════════════════════════════════════
# PROFESSIONAL COLOR SCHEME - Thomson Reuters inspired
# ═══════════════════════════════════════════════════════════════════════════════

COLORS = {
    # Primary palette
    'primary': '#2E5090',       # Deep professional blue
    'secondary': '#E85D04',     # Warm orange accent  
    'tertiary': '#457B9D',      # Soft steel blue
    
    # Backgrounds
    'bg_light': '#F8F9FA',      # Very light gray
    'bg_panel': '#EEF2F6',      # Light blue-gray panel
    'bg_accent': '#E3EBF2',     # Subtle blue tint
    
    # Grays
    'text_dark': '#2C3E50',     # Dark text
    'text_muted': '#6C757D',    # Muted text
    'grid': '#DEE2E6',          # Subtle grid lines
    'border': '#CED4DA',        # Borders
    
    # Gradient palette for bars (professional blues to teals)
    'gradient': ['#1A365D', '#2E5090', '#3D6BA8', '#4E8BC0', '#5FA8D8',
                 '#70C5F0', '#457B9D', '#5A9DBF', '#6FBFE1', '#84D1F3',
                 '#3D8B6F', '#4EA484', '#5FBD99', '#70D6AE', '#81EFC3'],
    
    # Vibrant palette for pie/donut charts (diverse, colorful but professional)
    'pie_colors': ['#2E5090', '#E85D04', '#2AA876', '#9B59B6', '#E74C3C',
                   '#3498DB', '#F39C12', '#1ABC9C', '#E91E63', '#00BCD4',
                   '#8BC34A', '#FF5722', '#607D8B', '#795548', '#9C27B0']
}


def setup_style():
    """Configure matplotlib/seaborn for professional, polished output."""
    # Reset to defaults first
    plt.rcdefaults()
    
    # Set seaborn theme with custom modifications
    sns.set_theme(style="whitegrid")
    
    # Professional rcParams
    plt.rcParams.update({
        # Figure
        'figure.figsize': (12, 6),
        'figure.facecolor': COLORS['bg_light'],
        'figure.edgecolor': COLORS['border'],
        'figure.dpi': 100,
        
        # Axes
        'axes.facecolor': COLORS['bg_panel'],
        'axes.edgecolor': COLORS['border'],
        'axes.linewidth': 0.8,
        'axes.titlesize': 14,
        'axes.titleweight': 'semibold',
        'axes.titlecolor': COLORS['text_dark'],
        'axes.labelsize': 11,
        'axes.labelcolor': COLORS['text_dark'],
        'axes.labelweight': 'medium',
        'axes.spines.top': False,
        'axes.spines.right': False,
        
        # Grid
        'axes.grid': True,
        'grid.color': COLORS['grid'],
        'grid.linewidth': 0.5,
        'grid.alpha': 0.7,
        
        # Ticks
        'xtick.color': COLORS['text_muted'],
        'ytick.color': COLORS['text_muted'],
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        
        # Font
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica Neue', 'Arial', 'DejaVu Sans'],
        'font.size': 10,
        
        # Legend
        'legend.frameon': True,
        'legend.facecolor': 'white',
        'legend.edgecolor': COLORS['border'],
        'legend.fontsize': 9,
        
        # Savefig
        'savefig.facecolor': COLORS['bg_light'],
        'savefig.edgecolor': COLORS['border'],
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.2,
    })


def _add_chart_branding(fig, subtitle: Optional[str] = None):
    """Add subtle professional branding to charts."""
    # Add a thin colored line at the top as accent
    fig.patches.append(mpatches.Rectangle(
        (0, 0.98), 1, 0.02, 
        transform=fig.transFigure, 
        facecolor=COLORS['primary'],
        edgecolor='none',
        zorder=10
    ))
    
    # Optional subtitle
    if subtitle:
        fig.text(0.5, 0.96, subtitle, ha='center', fontsize=9, 
                 color=COLORS['text_muted'], style='italic')


def plot_posture_distribution(
    stats: DatasetStatistics,
    top_n: int = 15,
    figsize: tuple = (14, 9),
    title: Optional[str] = None
) -> plt.Figure:
    """
    Creates a professional horizontal bar chart of posture label frequencies.
    
    Args:
        stats: Dataset statistics containing posture distribution
        top_n: Number of top postures to show
        figsize: Figure size tuple
        title: Optional custom title
    
    Returns:
        matplotlib Figure object
    """
    setup_style()
    
    # Get top N from the Series (already sorted by value_counts)
    top_postures = stats.posture_distribution.head(top_n)
    
    labels = list(top_postures.index)
    counts = list(top_postures.values)
    
    # Create figure with professional styling
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create gradient colors from primary palette
    n_colors = len(labels)
    colors = [COLORS['gradient'][i % len(COLORS['gradient'])] for i in range(n_colors)]
    
    # Horizontal bar chart with rounded appearance
    y_positions = range(len(labels))
    bars = ax.barh(y_positions, counts, color=colors, height=0.7, 
                   edgecolor='white', linewidth=0.5)
    
    # Customize axes
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()  # Highest at top
    ax.set_xlabel('Number of Documents', fontweight='medium', labelpad=10)
    
    # Professional title with padding
    ax.set_title(
        title or f'Top {top_n} Procedural Postures by Frequency',
        fontsize=16, fontweight='bold', color=COLORS['text_dark'],
        pad=20
    )
    
    # Add count labels on bars with professional styling
    max_count = max(counts)
    for bar, count in zip(bars, counts):
        # Position label inside or outside bar based on bar length
        x_pos = bar.get_width()
        if x_pos > max_count * 0.15:
            # Inside the bar
            ax.text(
                x_pos - max_count * 0.02,
                bar.get_y() + bar.get_height() / 2,
                f'{count:,}',
                va='center', ha='right',
                fontsize=9, fontweight='bold', color='white'
            )
        else:
            # Outside the bar
            ax.text(
                x_pos + max_count * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f'{count:,}',
                va='center', ha='left',
                fontsize=9, fontweight='medium', color=COLORS['text_dark']
            )
    
    # Add subtle shadow effect to bars
    for bar in bars:
        bar.set_zorder(2)
    
    # X-axis formatting
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    # Add branding accent
    _add_chart_branding(fig, subtitle='TR Data Challenge 2023 - Label Distribution Analysis')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def plot_text_length_distribution(
    word_counts: Union[list[int], pd.Series],
    figsize: tuple = (14, 6),
    bins: int = 50
) -> plt.Figure:
    """
    Creates professional histograms of document word counts.
    
    Args:
        word_counts: List or Series of word counts per document
        figsize: Figure size tuple
        bins: Number of histogram bins
    
    Returns:
        matplotlib Figure object
    """
    setup_style()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Ensure professional background styling
    fig.set_facecolor(COLORS['bg_light'])
    for ax in [ax1, ax2]:
        ax.set_facecolor(COLORS['bg_panel'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(COLORS['border'])
        ax.spines['bottom'].set_color(COLORS['border'])
    
    # Calculate statistics for annotation
    wc = pd.Series(word_counts)
    mean_val = wc.mean()
    median_val = wc.median()
    
    # Regular histogram with professional styling
    n1, bins1, patches1 = ax1.hist(word_counts, bins=bins, 
                                    color=COLORS['primary'], 
                                    edgecolor='white', 
                                    linewidth=0.5, 
                                    alpha=0.85)
    
    ax1.set_xlabel('Words per Document', fontweight='medium', labelpad=10)
    ax1.set_ylabel('Frequency', fontweight='medium', labelpad=10)
    ax1.set_title('Document Length Distribution', fontsize=13, fontweight='bold', 
                  color=COLORS['text_dark'], pad=15)
    
    # Add mean/median lines
    ax1.axvline(mean_val, color=COLORS['secondary'], linestyle='--', 
                linewidth=2, label=f'Mean: {mean_val:,.0f}')
    ax1.axvline(median_val, color=COLORS['tertiary'], linestyle='-', 
                linewidth=2, label=f'Median: {median_val:,.0f}')
    ax1.legend(loc='upper right', fontsize=9)
    
    # Log-scale histogram for better visibility of tail
    n2, bins2, patches2 = ax2.hist(word_counts, bins=bins, 
                                    color=COLORS['tertiary'], 
                                    edgecolor='white', 
                                    linewidth=0.5, 
                                    alpha=0.85)
    
    ax2.set_xlabel('Words per Document', fontweight='medium', labelpad=10)
    ax2.set_ylabel('Frequency (log scale)', fontweight='medium', labelpad=10)
    ax2.set_title('Document Length Distribution (Log Scale)', fontsize=13, 
                  fontweight='bold', color=COLORS['text_dark'], pad=15)
    ax2.set_yscale('log')
    
    # Add summary stats box
    stats_text = f'Documents: {len(wc):,}\nMin: {wc.min():,}\nMax: {wc.max():,}'
    ax2.text(0.97, 0.97, stats_text, transform=ax2.transAxes, fontsize=9,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                      edgecolor=COLORS['border'], alpha=0.9))
    
    # Add branding
    _add_chart_branding(fig, subtitle='Text Length Analysis')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def plot_class_imbalance(
    stats: DatasetStatistics,
    figsize: tuple = (14, 8)
) -> plt.Figure:
    """
    Creates a professional donut chart showing class distribution.
    Groups small classes into "Other" for readability.
    """
    setup_style()
    
    posture_dist = stats.posture_distribution
    
    # Top 8 + "Other"
    top_n = 8
    top_postures = posture_dist.head(top_n)
    other_count = posture_dist.iloc[top_n:].sum() if len(posture_dist) > top_n else 0
    
    labels = [str(p)[:35] + '...' if len(str(p)) > 35 else str(p) for p in top_postures.index]
    sizes = list(top_postures.values)
    
    if other_count > 0:
        labels.append(f'Other ({len(posture_dist) - top_n} classes)')
        sizes.append(other_count)
    
    # Vibrant color palette for pie chart (diverse colors)
    colors = [COLORS['pie_colors'][i % len(COLORS['pie_colors'])] for i in range(len(labels))]
    if other_count > 0:
        colors[-1] = COLORS['text_muted']  # Gray for "Other"
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, 
                                    gridspec_kw={'width_ratios': [1.2, 1]})
    
    # Ensure professional background styling
    fig.set_facecolor(COLORS['bg_light'])
    ax1.set_facecolor(COLORS['bg_light'])  # Pie chart looks better on light bg
    ax2.set_facecolor(COLORS['bg_panel'])
    ax2.axis('off')
    
    # Left: Donut chart
    wedges, texts, autotexts = ax1.pie(
        sizes,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        wedgeprops=dict(width=0.6, edgecolor='white', linewidth=2),
        pctdistance=0.75
    )
    
    # Style the percentage labels
    for autotext in autotexts:
        autotext.set_fontsize(9)
        autotext.set_fontweight('bold')
        autotext.set_color('white')
    
    # Add center text
    total = sum(sizes)
    ax1.text(0, 0, f'{total:,}\nTotal', ha='center', va='center', 
             fontsize=14, fontweight='bold', color=COLORS['text_dark'])
    
    ax1.set_title('Class Distribution Overview', fontsize=14, fontweight='bold',
                  color=COLORS['text_dark'], pad=20)
    
    # Right: Legend as a styled table
    ax2.axis('off')
    
    # Create legend data
    legend_data = list(zip(labels, sizes, [s/total*100 for s in sizes]))
    
    # Draw custom legend
    y_start = 0.95
    y_step = 0.085
    
    # Header
    ax2.text(0.05, y_start + 0.05, 'Posture Label', fontsize=11, fontweight='bold',
             color=COLORS['text_dark'], transform=ax2.transAxes)
    ax2.text(0.7, y_start + 0.05, 'Count', fontsize=11, fontweight='bold',
             color=COLORS['text_dark'], transform=ax2.transAxes, ha='center')
    ax2.text(0.9, y_start + 0.05, '%', fontsize=11, fontweight='bold',
             color=COLORS['text_dark'], transform=ax2.transAxes, ha='center')
    
    # Separator line (using plot instead of axhline for transform support)
    ax2.plot([0.02, 0.98], [y_start, y_start], color=COLORS['border'], 
             linewidth=1, transform=ax2.transAxes, clip_on=False)
    
    for i, (label, count, pct) in enumerate(legend_data):
        y_pos = y_start - (i + 1) * y_step
        
        # Color box
        ax2.add_patch(mpatches.Rectangle(
            (0.02, y_pos - 0.02), 0.025, 0.04,
            transform=ax2.transAxes,
            facecolor=colors[i],
            edgecolor='white',
            linewidth=1
        ))
        
        # Label (truncated if needed)
        display_label = label[:32] + '...' if len(label) > 32 else label
        ax2.text(0.06, y_pos, display_label, fontsize=9,
                 color=COLORS['text_dark'], transform=ax2.transAxes,
                 verticalalignment='center')
        
        # Count
        ax2.text(0.7, y_pos, f'{count:,}', fontsize=9, 
                 color=COLORS['text_dark'], transform=ax2.transAxes,
                 verticalalignment='center', ha='center')
        
        # Percentage
        ax2.text(0.9, y_pos, f'{pct:.1f}%', fontsize=9,
                 color=COLORS['text_dark'], transform=ax2.transAxes,
                 verticalalignment='center', ha='center')
        
        # Alternating row background
        if i % 2 == 0:
            ax2.add_patch(mpatches.Rectangle(
                (0.01, y_pos - 0.035), 0.97, 0.07,
                transform=ax2.transAxes,
                facecolor=COLORS['bg_accent'],
                edgecolor='none',
                zorder=-1
            ))
    
    # Add branding
    _add_chart_branding(fig, subtitle='Multi-Label Classification - Class Imbalance Analysis')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig
