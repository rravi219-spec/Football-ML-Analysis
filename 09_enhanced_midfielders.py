"""
ENHANCED MIDFIELDER CLASSIFICATION
Adds CM (Central Midfielders) and Box-to-Box classifications
Considers ALL seasons for career-average analysis
"""

import pandas as pd
import numpy as np

print("="*80)
print("⚙️ ENHANCED MIDFIELDER CLASSIFICATION")
print("Adding: CAM, CM, CDM, and Box-to-Box roles")
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
    'tackles': 'sum',
    'interceptions': 'sum',
    'progressive_passes': 'sum',
    'progressive_carries': 'sum',
    'passes_into_final_third': 'sum',
    'passes_into_penalty_area': 'sum',
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
player_aggregated['key_passes_per_90'] = player_aggregated['key_passes'] / player_aggregated['ninety_mins_played']
player_aggregated['tackles_per_90'] = player_aggregated['tackles'] / player_aggregated['ninety_mins_played']
player_aggregated['progressive_passes_per_90'] = player_aggregated['progressive_passes'] / player_aggregated['ninety_mins_played']
player_aggregated['progressive_actions_per_90'] = (player_aggregated['progressive_passes'] + player_aggregated['progressive_carries']) / player_aggregated['ninety_mins_played']
player_aggregated['final_third_involvement_per_90'] = (player_aggregated['passes_into_final_third'] + player_aggregated['passes_into_penalty_area']) / player_aggregated['ninety_mins_played']

# Filter minimum games
player_aggregated = player_aggregated[player_aggregated['min'] >= 1800].copy()

# ============================================================================
# MIDFIELDER SCORES
# ============================================================================

print("\n⚙️ Calculating midfielder-specific scores...")

# CAM SCORE (Attacking Midfielder - high creativity)
player_aggregated['cam_score'] = (
    (player_aggregated['key_passes_per_90'] / player_aggregated['key_passes_per_90'].max() * 50) +
    (player_aggregated['xa_per_90'] / player_aggregated['xa_per_90'].max() * 30) +
    (player_aggregated['final_third_involvement_per_90'] / player_aggregated['final_third_involvement_per_90'].max() * 20)
).clip(0, 100)

# CDM SCORE (Defensive Midfielder - high defensive actions)
player_aggregated['cdm_score'] = (
    (player_aggregated['tackles_per_90'] / player_aggregated['tackles_per_90'].max() * 60) +
    ((player_aggregated['interceptions'] / player_aggregated['ninety_mins_played']) / ((player_aggregated['interceptions'] / player_aggregated['ninety_mins_played']).max()) * 40)
).clip(0, 100)

# CM SCORE (Central Midfielder - balanced, high progressive passing)
player_aggregated['cm_score'] = (
    (player_aggregated['progressive_passes_per_90'] / player_aggregated['progressive_passes_per_90'].max() * 50) +
    (player_aggregated['progressive_actions_per_90'] / player_aggregated['progressive_actions_per_90'].max() * 30) +
    (player_aggregated['key_passes_per_90'] / player_aggregated['key_passes_per_90'].max() * 10) +
    (player_aggregated['tackles_per_90'] / player_aggregated['tackles_per_90'].max() * 10)
).clip(0, 100)

# BOX-TO-BOX SCORE (High in both attacking AND defensive metrics)
player_aggregated['box_to_box_score'] = (
    (player_aggregated['progressive_actions_per_90'] / player_aggregated['progressive_actions_per_90'].max() * 40) +
    (player_aggregated['tackles_per_90'] / player_aggregated['tackles_per_90'].max() * 30) +
    (player_aggregated['key_passes_per_90'] / player_aggregated['key_passes_per_90'].max() * 15) +
    ((player_aggregated['interceptions'] / player_aggregated['ninety_mins_played']) / ((player_aggregated['interceptions'] / player_aggregated['ninety_mins_played']).max()) * 15)
).clip(0, 100)

# ============================================================================
# CLASSIFY MIDFIELDERS
# ============================================================================

print("\n⚙️ Classifying midfielders...")

# Filter for midfielders only
midfielders = player_aggregated[
    player_aggregated['position'].str.contains('MF', na=False)
].copy()

# Exclude players who are primarily forwards/wingers (high goals)
midfielders = midfielders[midfielders['goals'] < 20].copy()

print(f"\n✓ Found {len(midfielders)} midfielders")

# ============================================================================
# ADVANCED CLASSIFICATION LOGIC
# ============================================================================

def classify_midfielder(row):
    """
    Classify midfielder based on multiple scores
    Priority: CAM > Box-to-Box > CDM > CM (default)
    """
    cam = row['cam_score']
    cdm = row['cdm_score']
    cm = row['cm_score']
    b2b = row['box_to_box_score']
    
    # CAM: High creativity, low defensive work
    if cam > 40 and cam > cdm + 15 and cam > cm + 10:
        return 'CAM'
    
    # Box-to-Box: High in BOTH attacking and defensive
    elif b2b > 45 and cdm > 40 and (cam > 25 or cm > 40):
        return 'BOX-TO-BOX'
    
    # CDM: High defensive, low attacking
    elif cdm > 50 and cdm > cam + 15:
        return 'CDM'
    
    # CM: Balanced, progressive
    elif cm > 35:
        return 'CM'
    
    # Default to CM if scores are balanced
    else:
        # If highest score is CDM but not dominant
        if cdm > cam and cdm > cm:
            return 'CDM'
        # If highest is CAM but not dominant
        elif cam > cdm and cam > cm:
            return 'CAM'
        # Default
        else:
            return 'CM'

midfielders['midfielder_role'] = midfielders.apply(classify_midfielder, axis=1)

# ============================================================================
# RESULTS
# ============================================================================

print("\n" + "="*80)
print("✅ MIDFIELDER CLASSIFICATION RESULTS")
print("="*80)

role_counts = midfielders['midfielder_role'].value_counts()
for role, count in role_counts.items():
    print(f"  {role}: {count}")

print(f"\nTotal midfielders: {len(midfielders)}")

# ============================================================================
# SAVE RESULTS
# ============================================================================

midfielders_sorted = midfielders.sort_values('cm_score', ascending=False)
midfielders_sorted.to_csv('/mnt/user-data/outputs/ENHANCED_MIDFIELDERS_ALL_ROLES.csv', index=False)

print("\n💾 Saved: ENHANCED_MIDFIELDERS_ALL_ROLES.csv")

# ============================================================================
# TOP 10 FOR EACH ROLE
# ============================================================================

print("\n" + "="*80)
print("🏆 TOP 10 PLAYERS BY MIDFIELDER ROLE (CAREER AVERAGE)")
print("="*80)

# CAM
print("\n" + "="*80)
print("🎨 TOP 10 ATTACKING MIDFIELDERS (CAM)")
print("="*80)

top_cam = midfielders[midfielders['midfielder_role'] == 'CAM'].nlargest(10, 'cam_score')
for i, (_, row) in enumerate(top_cam.iterrows(), 1):
    print(f"{i:2}. {row['player']:30} ({row['squad']:25}) | Score: {row['cam_score']:.1f} | KP/90: {row['key_passes_per_90']:.2f} | {row['seasons_played']} seasons")

# CM
print("\n" + "="*80)
print("⚙️ TOP 10 CENTRAL MIDFIELDERS (CM)")
print("="*80)

top_cm = midfielders[midfielders['midfielder_role'] == 'CM'].nlargest(10, 'cm_score')
for i, (_, row) in enumerate(top_cm.iterrows(), 1):
    print(f"{i:2}. {row['player']:30} ({row['squad']:25}) | Score: {row['cm_score']:.1f} | Prog/90: {row['progressive_actions_per_90']:.2f} | {row['seasons_played']} seasons")

# CDM
print("\n" + "="*80)
print("🛡️ TOP 10 DEFENSIVE MIDFIELDERS (CDM)")
print("="*80)

top_cdm = midfielders[midfielders['midfielder_role'] == 'CDM'].nlargest(10, 'cdm_score')
for i, (_, row) in enumerate(top_cdm.iterrows(), 1):
    print(f"{i:2}. {row['player']:30} ({row['squad']:25}) | Score: {row['cdm_score']:.1f} | Tackles/90: {row['tackles_per_90']:.2f} | {row['seasons_played']} seasons")

# Box-to-Box
print("\n" + "="*80)
print("💪 TOP 10 BOX-TO-BOX MIDFIELDERS")
print("="*80)

top_b2b = midfielders[midfielders['midfielder_role'] == 'BOX-TO-BOX'].nlargest(10, 'box_to_box_score')
for i, (_, row) in enumerate(top_b2b.iterrows(), 1):
    print(f"{i:2}. {row['player']:30} ({row['squad']:25}) | Score: {row['box_to_box_score']:.1f} | Prog/90: {row['progressive_actions_per_90']:.2f} | T/90: {row['tackles_per_90']:.2f} | {row['seasons_played']} seasons")

print("\n" + "="*80)
print("✅ COMPLETE! Enhanced midfielder classification with all 4 roles!")
print("="*80)

