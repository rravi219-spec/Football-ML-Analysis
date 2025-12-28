"""
ENHANCED WINGER CLASSIFICATION
Separates wingers into LW (Left Wing) and RW (Right Wing)
Uses statistical inference based on progressive carries and crossing patterns
"""

import pandas as pd
import numpy as np

print("="*80)
print("⚡ ENHANCED WINGER CLASSIFICATION")
print("Separating wingers into LW (Left Wing) and RW (Right Wing)")
print("="*80)

# Load master dataset
df = pd.read_csv('/mnt/user-data/outputs/big5_leagues_WITH_TRACKING_BACK.csv')

print(f"\n📊 Dataset: {len(df):,} records")

# ============================================================================
# AGGREGATE BY PLAYER (ALL SEASONS)
# ============================================================================

print("\n📈 Aggregating player statistics across all seasons...")

player_aggregated = df.groupby('player').agg({
    'min': 'sum',
    'goals': 'sum',
    'assists': 'sum',
    'xg': 'sum',
    'xa': 'sum',
    'shots': 'sum',
    'key_passes': 'sum',
    'progressive_passes': 'sum',
    'progressive_carries': 'sum',
    'passes_into_final_third': 'sum',
    'crosses_into_penalty_area': 'sum',
    'season': 'count',
    'squad': 'last',
    'comp': 'last',
    'position': 'last',
    'age': 'last'
}).reset_index()

player_aggregated.rename(columns={'season': 'seasons_played'}, inplace=True)

# Calculate per 90 stats
player_aggregated['ninety_mins_played'] = player_aggregated['min'] / 90
player_aggregated['goals_per_90'] = player_aggregated['goals'] / player_aggregated['ninety_mins_played']
player_aggregated['assists_per_90'] = player_aggregated['assists'] / player_aggregated['ninety_mins_played']
player_aggregated['xg_per_90'] = player_aggregated['xg'] / player_aggregated['ninety_mins_played']
player_aggregated['xa_per_90'] = player_aggregated['xa'] / player_aggregated['ninety_mins_played']
player_aggregated['progressive_carries_per_90'] = player_aggregated['progressive_carries'] / player_aggregated['ninety_mins_played']
player_aggregated['crosses_per_90'] = player_aggregated['crosses_into_penalty_area'] / player_aggregated['ninety_mins_played']

# Filter minimum games
player_aggregated = player_aggregated[player_aggregated['min'] >= 1800].copy()

# ============================================================================
# WINGER SCORE
# ============================================================================

print("\n⚡ Calculating winger scores...")

player_aggregated['winger_score'] = (
    (player_aggregated['goals_per_90'] / player_aggregated['goals_per_90'].max() * 30) +
    (player_aggregated['assists_per_90'] / player_aggregated['assists_per_90'].max() * 30) +
    (player_aggregated['progressive_carries_per_90'] / player_aggregated['progressive_carries_per_90'].max() * 25) +
    (player_aggregated['crosses_per_90'] / player_aggregated['crosses_per_90'].max() * 15)
).clip(0, 100)

# ============================================================================
# IDENTIFY WINGERS
# ============================================================================

print("\n⚡ Identifying wingers...")

# Wingers: FW/MF with high progressive carries, not pure strikers
wingers = player_aggregated[
    (player_aggregated['position'].str.contains('FW|MF', na=False)) &
    (player_aggregated['progressive_carries'] >= 80) &
    (player_aggregated['crosses_into_penalty_area'] >= 5) &
    (player_aggregated['goals'] < 25)  # Not pure strikers
].copy()

print(f"\n✓ Found {len(wingers)} wingers")

# ============================================================================
# CLASSIFY LW vs RW USING STATISTICAL INFERENCE
# ============================================================================

print("\n⚡ Classifying LW vs RW using statistical inference...")

def classify_winger_side(wingers_df):
    """
    Classify wingers as LW or RW using within-squad comparative analysis
    
    Logic:
    - Within each squad, wingers are sorted by progressive carries
    - Higher progressive carriers tend to be RW (cutting inside right-to-left)
    - Lower carriers tend to be LW
    - Alternating assignment for multiple wingers per squad
    """
    results = []
    
    for squad in wingers_df['squad'].unique():
        squad_wingers = wingers_df[wingers_df['squad'] == squad].copy()
        
        if len(squad_wingers) == 0:
            continue
        elif len(squad_wingers) == 1:
            # Single winger - use crosses as indicator
            player = squad_wingers.iloc[0]
            # Higher crossers tend to be traditional wingers (RW)
            if player['crosses_per_90'] > wingers_df['crosses_per_90'].median():
                squad_wingers.loc[squad_wingers.index[0], 'winger_position'] = 'RW'
            else:
                squad_wingers.loc[squad_wingers.index[0], 'winger_position'] = 'LW'
        else:
            # Multiple wingers - sort and alternate
            squad_wingers = squad_wingers.sort_values('progressive_carries_per_90', ascending=False)
            
            for i, idx in enumerate(squad_wingers.index):
                # Alternate assignment
                if i % 2 == 0:
                    squad_wingers.loc[idx, 'winger_position'] = 'RW'
                else:
                    squad_wingers.loc[idx, 'winger_position'] = 'LW'
        
        results.append(squad_wingers)
    
    if results:
        return pd.concat(results, ignore_index=False)
    else:
        return wingers_df

wingers = classify_winger_side(wingers)

# For any unclassified, use progressive carries as default
if wingers['winger_position'].isna().any():
    wingers.loc[wingers['winger_position'].isna(), 'winger_position'] = wingers[wingers['winger_position'].isna()].apply(
        lambda x: 'RW' if x['progressive_carries_per_90'] > wingers['progressive_carries_per_90'].median() else 'LW',
        axis=1
    )

# ============================================================================
# RESULTS
# ============================================================================

print("\n" + "="*80)
print("✅ WINGER CLASSIFICATION RESULTS")
print("="*80)

position_counts = wingers['winger_position'].value_counts()
for pos, count in position_counts.items():
    print(f"  {pos}: {count}")

print(f"\nTotal wingers: {len(wingers)}")

# ============================================================================
# SAVE RESULTS
# ============================================================================

wingers_sorted = wingers.sort_values('winger_score', ascending=False)
wingers_sorted.to_csv('/mnt/user-data/outputs/ENHANCED_WINGERS_LW_RW.csv', index=False)

print("\n💾 Saved: ENHANCED_WINGERS_LW_RW.csv")

# ============================================================================
# TOP 10 FOR EACH POSITION
# ============================================================================

print("\n" + "="*80)
print("🏆 TOP 10 WINGERS BY POSITION (CAREER AVERAGE)")
print("="*80)

# LEFT WING
print("\n" + "="*80)
print("⬅️ TOP 10 LEFT WINGERS (LW)")
print("="*80)

top_lw = wingers[wingers['winger_position'] == 'LW'].nlargest(10, 'winger_score')
for i, (_, row) in enumerate(top_lw.iterrows(), 1):
    print(f"{i:2}. {row['player']:30} ({row['squad']:25}) | Score: {row['winger_score']:.1f} | G/90: {row['goals_per_90']:.2f} | A/90: {row['assists_per_90']:.2f} | {row['seasons_played']} seasons")

# RIGHT WING
print("\n" + "="*80)
print("➡️ TOP 10 RIGHT WINGERS (RW)")
print("="*80)

top_rw = wingers[wingers['winger_position'] == 'RW'].nlargest(10, 'winger_score')
for i, (_, row) in enumerate(top_rw.iterrows(), 1):
    print(f"{i:2}. {row['player']:30} ({row['squad']:25}) | Score: {row['winger_score']:.1f} | G/90: {row['goals_per_90']:.2f} | A/90: {row['assists_per_90']:.2f} | {row['seasons_played']} seasons")

# ALL WINGERS (Top 20 overall)
print("\n" + "="*80)
print("⚡ TOP 20 WINGERS OVERALL (LW + RW)")
print("="*80)

top_all = wingers.nlargest(20, 'winger_score')
for i, (_, row) in enumerate(top_all.iterrows(), 1):
    print(f"{i:2}. {row['player']:30} ({row['squad']:25}) [{row['winger_position']}] | Score: {row['winger_score']:.1f} | G/90: {row['goals_per_90']:.2f} | A/90: {row['assists_per_90']:.2f}")

print("\n" + "="*80)
print("✅ COMPLETE! Enhanced winger classification with LW/RW distinction!")
print("="*80)

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

print("\n📊 WINGER STATISTICS BY POSITION:")
print("\nLEFT WINGERS (LW):")
lw_stats = wingers[wingers['winger_position'] == 'LW']
print(f"  Count: {len(lw_stats)}")
print(f"  Average Goals/90: {lw_stats['goals_per_90'].mean():.2f}")
print(f"  Average Assists/90: {lw_stats['assists_per_90'].mean():.2f}")
print(f"  Average Winger Score: {lw_stats['winger_score'].mean():.1f}")
print(f"  Average Progressive Carries/90: {lw_stats['progressive_carries_per_90'].mean():.2f}")

print("\nRIGHT WINGERS (RW):")
rw_stats = wingers[wingers['winger_position'] == 'RW']
print(f"  Count: {len(rw_stats)}")
print(f"  Average Goals/90: {rw_stats['goals_per_90'].mean():.2f}")
print(f"  Average Assists/90: {rw_stats['assists_per_90'].mean():.2f}")
print(f"  Average Winger Score: {rw_stats['winger_score'].mean():.1f}")
print(f"  Average Progressive Carries/90: {rw_stats['progressive_carries_per_90'].mean():.2f}")

print("\n" + "="*80)

