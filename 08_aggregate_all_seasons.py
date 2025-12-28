"""
MULTI-SEASON AGGREGATE PLAYER ANALYSIS
Considers ALL seasons (2018-2024) for each player
Shows career-average performance, not just current season
"""

import pandas as pd
import numpy as np

print("="*80)
print("🏆 MULTI-SEASON AGGREGATE PLAYER ANALYSIS")
print("Considering ALL seasons (2018-2024) for each player")
print("="*80)

# Load master dataset
df = pd.read_csv('/mnt/user-data/outputs/big5_leagues_WITH_TRACKING_BACK.csv')

print(f"\n📊 Dataset: {len(df):,} records, {df['player'].nunique():,} unique players")
print(f"Seasons: {sorted(df['season'].unique())}")

# ============================================================================
# AGGREGATE BY PLAYER (ALL SEASONS COMBINED)
# ============================================================================

print("\n" + "="*80)
print("📈 AGGREGATING PLAYER STATISTICS ACROSS ALL SEASONS...")
print("="*80)

# Group by player and aggregate
player_aggregated = df.groupby('player').agg({
    'min': 'sum',  # Total minutes
    'goals': 'sum',  # Total goals
    'assists': 'sum',  # Total assists
    'xg': 'sum',  # Total xG
    'xa': 'sum',  # Total xA
    'shots': 'sum',  # Total shots
    'key_passes': 'sum',  # Total key passes
    'tackles': 'sum',  # Total tackles
    'interceptions': 'sum',  # Total interceptions
    'clearances': 'sum',  # Total clearances
    'progressive_passes': 'sum',
    'progressive_carries': 'sum',
    'passes_into_final_third': 'sum',
    'crosses_into_penalty_area': 'sum',
    'tracking_back_tackles': 'sum',
    'defensive_work_rate': 'sum',
    'season': 'count',  # Number of seasons
    'squad': 'last',  # Latest squad
    'comp': 'last',  # Latest competition
    'position': 'last',  # Latest position
    'age': 'last'  # Latest age
}).reset_index()

# Rename season count
player_aggregated.rename(columns={'season': 'seasons_played'}, inplace=True)

# Calculate per 90 stats
player_aggregated['ninety_mins_played'] = player_aggregated['min'] / 90
player_aggregated['goals_per_90'] = player_aggregated['goals'] / player_aggregated['ninety_mins_played']
player_aggregated['assists_per_90'] = player_aggregated['assists'] / player_aggregated['ninety_mins_played']
player_aggregated['xg_per_90'] = player_aggregated['xg'] / player_aggregated['ninety_mins_played']
player_aggregated['xa_per_90'] = player_aggregated['xa'] / player_aggregated['ninety_mins_played']
player_aggregated['shots_per_90'] = player_aggregated['shots'] / player_aggregated['ninety_mins_played']
player_aggregated['key_passes_per_90'] = player_aggregated['key_passes'] / player_aggregated['ninety_mins_played']
player_aggregated['tackles_per_90'] = player_aggregated['tackles'] / player_aggregated['ninety_mins_played']
player_aggregated['defensive_actions_per_90'] = (player_aggregated['tackles'] + player_aggregated['interceptions']) / player_aggregated['ninety_mins_played']

# Calculate aggregate scores
# POACHER SCORE (based on goals + xG)
player_aggregated['poacher_score'] = (
    (player_aggregated['goals_per_90'] / player_aggregated['goals_per_90'].max() * 50) +
    (player_aggregated['xg_per_90'] / player_aggregated['xg_per_90'].max() * 50)
).clip(0, 100)

# FALSE 9 SCORE (based on assists + key passes)
player_aggregated['false9_score'] = (
    (player_aggregated['assists_per_90'] / player_aggregated['assists_per_90'].max() * 50) +
    (player_aggregated['key_passes_per_90'] / player_aggregated['key_passes_per_90'].max() * 50)
).clip(0, 100)

# WINGER SCORE (goals + assists + progressive actions)
player_aggregated['winger_score'] = (
    (player_aggregated['goals_per_90'] / player_aggregated['goals_per_90'].max() * 35) +
    (player_aggregated['assists_per_90'] / player_aggregated['assists_per_90'].max() * 35) +
    (player_aggregated['progressive_carries'] / player_aggregated['progressive_carries'].max() * 30)
).clip(0, 100)

# CAM SCORE (key passes + xA)
player_aggregated['cam_score'] = (
    (player_aggregated['key_passes_per_90'] / player_aggregated['key_passes_per_90'].max() * 60) +
    (player_aggregated['xa_per_90'] / player_aggregated['xa_per_90'].max() * 40)
).clip(0, 100)

# CDM SCORE (tackles + interceptions)
player_aggregated['cdm_score'] = (
    (player_aggregated['tackles_per_90'] / player_aggregated['tackles_per_90'].max() * 50) +
    ((player_aggregated['interceptions'] / player_aggregated['ninety_mins_played']) / ((player_aggregated['interceptions'] / player_aggregated['ninety_mins_played']).max()) * 50)
).clip(0, 100)

# DEFENDER SCORE (clearances + tackles + interceptions)
player_aggregated['defender_score'] = (
    (player_aggregated['clearances'] / player_aggregated['clearances'].max() * 40) +
    (player_aggregated['tackles'] / player_aggregated['tackles'].max() * 30) +
    (player_aggregated['interceptions'] / player_aggregated['interceptions'].max() * 30)
).clip(0, 100)

# Filter minimum games (at least 20 full matches worth of minutes = 1800 mins)
player_aggregated = player_aggregated[player_aggregated['min'] >= 1800].copy()

print(f"\n✅ Aggregated {len(player_aggregated):,} players with 1800+ minutes across all seasons")

# ============================================================================
# CLASSIFY PLAYERS BY POSITION
# ============================================================================

print("\n" + "="*80)
print("⚽ CLASSIFYING PLAYERS INTO ROLES...")
print("="*80)

# FORWARDS (FW position + high goals)
forwards = player_aggregated[
    (player_aggregated['position'].str.contains('FW', na=False)) &
    (player_aggregated['goals'] >= 15)
].copy()

# Classify strikers
forwards['striker_role'] = forwards.apply(
    lambda x: 'POACHER' if x['poacher_score'] > x['false9_score'] else 'FALSE 9',
    axis=1
)

print(f"✅ {len(forwards)} Forwards classified")
print(f"   - Poachers: {len(forwards[forwards['striker_role'] == 'POACHER'])}")
print(f"   - False 9s: {len(forwards[forwards['striker_role'] == 'FALSE 9'])}")

# WINGERS (FW/MF with high progressive actions)
wingers = player_aggregated[
    (player_aggregated['position'].str.contains('FW|MF', na=False)) &
    (player_aggregated['progressive_carries'] >= 100) &
    (~player_aggregated.index.isin(forwards.index))
].copy()

wingers['winger_score_calc'] = wingers['winger_score']
print(f"✅ {len(wingers)} Wingers classified")

# MIDFIELDERS (MF position)
midfielders = player_aggregated[
    (player_aggregated['position'].str.contains('MF', na=False)) &
    (~player_aggregated.index.isin(forwards.index)) &
    (~player_aggregated.index.isin(wingers.index))
].copy()

# Classify midfielders
midfielders['midfielder_role'] = midfielders.apply(
    lambda x: 'CAM' if x['cam_score'] > x['cdm_score'] else 'CDM',
    axis=1
)

print(f"✅ {len(midfielders)} Midfielders classified")
print(f"   - CAMs: {len(midfielders[midfielders['midfielder_role'] == 'CAM'])}")
print(f"   - CDMs: {len(midfielders[midfielders['midfielder_role'] == 'CDM'])}")

# DEFENDERS (DF position)
defenders = player_aggregated[
    (player_aggregated['position'].str.contains('DF', na=False))
].copy()

# Simple CB classification (high clearances)
defenders['defender_type'] = defenders.apply(
    lambda x: 'CB' if x['clearances'] >= 70 else 'FB/WB',
    axis=1
)

print(f"✅ {len(defenders)} Defenders classified")
print(f"   - CBs: {len(defenders[defenders['defender_type'] == 'CB'])}")
print(f"   - FB/WB: {len(defenders[defenders['defender_type'] == 'FB/WB'])}")

# ============================================================================
# SAVE AGGREGATED CLASSIFICATIONS
# ============================================================================

# Save to outputs
forwards.to_csv('/mnt/user-data/outputs/MULTI_SEASON_STRIKERS.csv', index=False)
wingers.to_csv('/mnt/user-data/outputs/MULTI_SEASON_WINGERS.csv', index=False)
midfielders.to_csv('/mnt/user-data/outputs/MULTI_SEASON_MIDFIELDERS.csv', index=False)
defenders.to_csv('/mnt/user-data/outputs/MULTI_SEASON_DEFENDERS.csv', index=False)

print("\n" + "="*80)
print("💾 SAVED MULTI-SEASON CLASSIFICATIONS")
print("="*80)
print("✅ MULTI_SEASON_STRIKERS.csv")
print("✅ MULTI_SEASON_WINGERS.csv")
print("✅ MULTI_SEASON_MIDFIELDERS.csv")
print("✅ MULTI_SEASON_DEFENDERS.csv")

# ============================================================================
# TOP 10 RANKINGS - ALL SEASONS
# ============================================================================

print("\n" + "="*80)
print("🏆 TOP 10 PLAYERS (MULTI-SEASON CAREER STATS)")
print("="*80)

print("\n" + "="*80)
print("⚽ TOP 10 POACHERS (Career Average)")
print("="*80)

top_poachers = forwards[forwards['striker_role'] == 'POACHER'].nlargest(10, 'poacher_score')
for i, (_, row) in enumerate(top_poachers.iterrows(), 1):
    print(f"{i:2}. {row['player']:30} ({row['squad']:25}) | Score: {row['poacher_score']:.1f} | Goals/90: {row['goals_per_90']:.2f} | {row['seasons_played']} seasons")

print("\n" + "="*80)
print("🎨 TOP 10 FALSE 9s (Career Average)")
print("="*80)

top_false9s = forwards[forwards['striker_role'] == 'FALSE 9'].nlargest(10, 'false9_score')
for i, (_, row) in enumerate(top_false9s.iterrows(), 1):
    print(f"{i:2}. {row['player']:30} ({row['squad']:25}) | Score: {row['false9_score']:.1f} | Assists/90: {row['assists_per_90']:.2f} | {row['seasons_played']} seasons")

print("\n" + "="*80)
print("⚡ TOP 10 WINGERS (Career Average)")
print("="*80)

top_wingers = wingers.nlargest(10, 'winger_score')
for i, (_, row) in enumerate(top_wingers.iterrows(), 1):
    print(f"{i:2}. {row['player']:30} ({row['squad']:25}) | Score: {row['winger_score']:.1f} | G/90: {row['goals_per_90']:.2f} | A/90: {row['assists_per_90']:.2f} | {row['seasons_played']} seasons")

print("\n" + "="*80)
print("🎨 TOP 10 ATTACKING MIDFIELDERS (Career Average)")
print("="*80)

top_cams = midfielders.nlargest(10, 'cam_score')
for i, (_, row) in enumerate(top_cams.iterrows(), 1):
    print(f"{i:2}. {row['player']:30} ({row['squad']:25}) | Score: {row['cam_score']:.1f} | KP/90: {row['key_passes_per_90']:.2f} | {row['seasons_played']} seasons")

print("\n" + "="*80)
print("🛡️ TOP 10 DEFENSIVE MIDFIELDERS (Career Average)")
print("="*80)

top_cdms = midfielders.nlargest(10, 'cdm_score')
for i, (_, row) in enumerate(top_cdms.iterrows(), 1):
    print(f"{i:2}. {row['player']:30} ({row['squad']:25}) | Score: {row['cdm_score']:.1f} | Tackles/90: {row['tackles_per_90']:.2f} | {row['seasons_played']} seasons")

print("\n" + "="*80)
print("🏰 TOP 10 DEFENDERS (Career Average)")
print("="*80)

top_defenders = defenders.nlargest(10, 'defender_score')
for i, (_, row) in enumerate(top_defenders.iterrows(), 1):
    print(f"{i:2}. {row['player']:30} ({row['squad']:25}) | Score: {row['defender_score']:.1f} | Type: {row['defender_type']} | {row['seasons_played']} seasons")

print("\n" + "="*80)
print("✅ COMPLETE! Multi-season analysis done!")
print("="*80)
print(f"\nTotal classified: {len(forwards) + len(wingers) + len(midfielders) + len(defenders)} players")
print(f"  - Forwards: {len(forwards)}")
print(f"  - Wingers: {len(wingers)}")
print(f"  - Midfielders: {len(midfielders)}")
print(f"  - Defenders: {len(defenders)}")

