"""
ADVANCED PLAYER ROLE IDENTIFICATION
Identifying specific roles within each position:
- Forwards: Poacher vs False 9
- Midfielders: CAM vs CDM vs Box-to-Box
- Defenders: RB, LB, CB, RWB, LWB
"""

import pandas as pd
import numpy as np

print("="*80)
print("🎯 ADVANCED PLAYER ROLE IDENTIFICATION")
print("="*80)

# Load dataset
print("\n📂 Loading dataset...")
df = pd.read_csv('/mnt/user-data/outputs/big5_leagues_WITH_TRACKING_BACK.csv')
print(f"✓ Loaded {len(df)} records")

# Filter for 2024 season only (most recent)
df_2024 = df[df['season'] == 2024].copy()
print(f"✓ Filtered to 2024 season: {len(df_2024)} records")

# ============================================================================
# PART 1: FORWARD ROLE IDENTIFICATION (POACHER VS FALSE 9)
# ============================================================================
print("\n" + "="*80)
print("⚡ PART 1: FORWARD ROLE IDENTIFICATION")
print("="*80)

forwards = df_2024[df_2024['general_position'] == 'Forward'].copy()
print(f"\nForwards in 2024: {len(forwards)}")

# Define Poacher characteristics:
# - High goals per 90
# - High shot efficiency
# - High poacher index
# - Low key passes (not creating)
# - Low progressive passes (not dropping deep)

# Define False 9 characteristics:
# - Moderate goals
# - High key passes (creating)
# - High progressive passes received (dropping deep)
# - High assists
# - High complete forward index

if len(forwards) > 0:
    # Normalize metrics for scoring
    forwards['goals_norm'] = (forwards['goals_per_90'] / forwards['goals_per_90'].max()) * 100
    forwards['shot_eff_norm'] = (forwards['shot_efficiency'] / forwards['shot_efficiency'].max()) * 100
    forwards['key_pass_norm'] = (forwards['key_passes'] / forwards['key_passes'].max()) * 100
    forwards['assists_norm'] = (forwards['assists_per_90'] / forwards['assists_per_90'].max()) * 100
    forwards['prog_recv_norm'] = (forwards['progressive_passes_received'] / 
                                   forwards['progressive_passes_received'].max()) * 100
    
    # POACHER SCORE (0-100)
    # High: goals, shot efficiency, poacher index
    # Low penalty: key passes (they don't create much)
    forwards['poacher_score'] = (
        forwards['goals_norm'] * 0.35 +
        forwards['shot_eff_norm'] * 0.25 +
        forwards['poacher_index'] * 0.30 +
        (100 - forwards['key_pass_norm']) * 0.10  # Penalty for creating
    )
    
    # FALSE 9 SCORE (0-100)
    # High: key passes, assists, progressive passes received, complete forward index
    # Moderate: goals (they still score, just not as much)
    forwards['false9_score'] = (
        forwards['key_pass_norm'] * 0.30 +
        forwards['assists_norm'] * 0.25 +
        forwards['prog_recv_norm'] * 0.20 +
        forwards['complete_forward_index'] * 0.25
    )
    
    # Classify role (whichever score is higher)
    forwards['forward_role'] = forwards.apply(
        lambda x: 'POACHER' if x['poacher_score'] > x['false9_score'] else 'FALSE 9',
        axis=1
    )
    
    # Add role strength (how pure is the role)
    forwards['role_strength'] = abs(forwards['poacher_score'] - forwards['false9_score'])
    
    print(f"\n📊 Forward Role Distribution:")
    print(forwards['forward_role'].value_counts())
    
    # TOP POACHERS
    print(f"\n🎯 TOP 10 PURE POACHERS (2024):")
    print("-" * 80)
    top_poachers = forwards.nlargest(10, 'poacher_score')[
        ['player', 'squad', 'comp', 'poacher_score', 'goals_per_90', 'shot_efficiency', 'forward_role']
    ]
    print(top_poachers.to_string(index=False))
    
    # TOP FALSE 9s
    print(f"\n🎨 TOP 10 FALSE 9 FORWARDS (2024):")
    print("-" * 80)
    top_false9 = forwards.nlargest(10, 'false9_score')[
        ['player', 'squad', 'comp', 'false9_score', 'key_passes', 'assists_per_90', 'forward_role']
    ]
    print(top_false9.to_string(index=False))

# ============================================================================
# PART 2: MIDFIELDER ROLE IDENTIFICATION (CAM, CDM, BOX-TO-BOX)
# ============================================================================
print("\n" + "="*80)
print("⚙️ PART 2: MIDFIELDER ROLE IDENTIFICATION")
print("="*80)

midfielders = df_2024[df_2024['general_position'] == 'Midfielder'].copy()
print(f"\nMidfielders in 2024: {len(midfielders)}")

# CAM (Attacking Midfielder):
# - High attacking_midfielder_index
# - High key passes
# - High final third involvement
# - High xAG

# CDM (Defensive Midfielder):
# - High defensive_midfielder_index
# - High tackles + interceptions
# - High pass completion
# - Low attacking involvement

# Box-to-Box:
# - High box_to_box_midfielder_score
# - Balanced attacking + defensive stats
# - High progressive passes
# - Moderate tackles

if len(midfielders) > 0:
    # Use our already-calculated indices + boost with specific metrics
    
    # CAM SCORE
    midfielders['cam_score'] = (
        midfielders['attacking_midfielder_index'] * 0.50 +
        (midfielders['final_third_involvement_per_90'] / 
         midfielders['final_third_involvement_per_90'].max() * 100) * 0.30 +
        (midfielders['creative_output_per_90'] / 
         midfielders['creative_output_per_90'].max() * 100) * 0.20
    )
    
    # CDM SCORE
    midfielders['cdm_score'] = (
        midfielders['defensive_midfielder_index'] * 0.50 +
        (midfielders['defensive_actions_per_90'] / 
         midfielders['defensive_actions_per_90'].max() * 100) * 0.30 +
        (midfielders['pass_completion_pct'] / 100 * 100) * 0.20
    )
    
    # BOX-TO-BOX SCORE (already have this!)
    midfielders['b2b_score'] = midfielders['box_to_box_midfielder_score']
    
    # Classify primary role (highest score)
    def classify_midfielder(row):
        scores = {
            'CAM': row['cam_score'],
            'CDM': row['cdm_score'],
            'BOX-TO-BOX': row['b2b_score']
        }
        return max(scores, key=scores.get)
    
    midfielders['midfielder_role'] = midfielders.apply(classify_midfielder, axis=1)
    
    # Role strength (gap between highest and second-highest)
    def role_strength(row):
        scores = [row['cam_score'], row['cdm_score'], row['b2b_score']]
        scores.sort(reverse=True)
        return scores[0] - scores[1]
    
    midfielders['role_strength'] = midfielders.apply(role_strength, axis=1)
    
    print(f"\n📊 Midfielder Role Distribution:")
    print(midfielders['midfielder_role'].value_counts())
    
    # TOP CAMs
    print(f"\n🎨 TOP 10 CAMs (Attacking Midfielders - 2024):")
    print("-" * 80)
    top_cams = midfielders.nlargest(10, 'cam_score')[
        ['player', 'squad', 'comp', 'cam_score', 'key_passes', 'final_third_involvement_per_90', 'midfielder_role']
    ]
    print(top_cams.to_string(index=False))
    
    # TOP CDMs
    print(f"\n🛡️ TOP 10 CDMs (Defensive Midfielders - 2024):")
    print("-" * 80)
    top_cdms = midfielders.nlargest(10, 'cdm_score')[
        ['player', 'squad', 'comp', 'cdm_score', 'tackles', 'interceptions', 'midfielder_role']
    ]
    print(top_cdms.to_string(index=False))
    
    # TOP BOX-TO-BOX
    print(f"\n⚡ TOP 10 BOX-TO-BOX MIDFIELDERS (2024):")
    print("-" * 80)
    top_b2b = midfielders.nlargest(10, 'b2b_score')[
        ['player', 'squad', 'comp', 'b2b_score', 'progressive_passes', 'tackles', 'midfielder_role']
    ]
    print(top_b2b.to_string(index=False))

# ============================================================================
# PART 3: DEFENDER POSITION IDENTIFICATION (RB, LB, CB, RWB, LWB)
# ============================================================================
print("\n" + "="*80)
print("🛡️ PART 3: DEFENDER POSITION IDENTIFICATION")
print("="*80)

defenders = df_2024[df_2024['general_position'] == 'Defender'].copy()
print(f"\nDefenders in 2024: {len(defenders)}")

# Check if we have position data in the original 'position' column
print(f"\n📋 Checking position details...")
if 'position' in defenders.columns:
    print(f"Position column values (sample):")
    print(defenders['position'].value_counts().head(20))

# STRATEGY:
# Use the 'position' column to identify FB (fullback), CB (center back), WB (wing back)
# Then use stats to determine if RB/LB or RWB/LWB

if 'position' in defenders.columns:
    # Clean position column
    defenders['pos_clean'] = defenders['position'].str.upper()
    
    # Identify position types
    def identify_defender_position(row):
        pos = str(row['pos_clean'])
        
        # Center Backs
        if 'CB' in pos or 'DF' in pos:
            return 'CB'
        
        # Fullbacks / Wing Backs
        elif 'FB' in pos or 'WB' in pos or 'LB' in pos or 'RB' in pos:
            # Use stats to determine if attacking (wing back) or defensive (fullback)
            attacking_score = (
                row['attacking_fullback_index'] +
                row['crosses_into_penalty_area'] +
                row['progressive_carries']
            )
            
            # If high attacking involvement = wing back, else fullback
            is_wingback = attacking_score > defenders['attacking_fullback_index'].median() * 2
            
            # Determine side (left or right)
            if 'L' in pos or 'LEFT' in pos:
                return 'LWB' if is_wingback else 'LB'
            elif 'R' in pos or 'RIGHT' in pos:
                return 'RWB' if is_wingback else 'RB'
            else:
                # Use crosses to guess side (more sophisticated clubs track this)
                return 'RWB/LWB' if is_wingback else 'RB/LB'
        
        else:
            return 'CB'  # Default to CB if unclear
    
    defenders['defender_position'] = defenders.apply(identify_defender_position, axis=1)
    
    print(f"\n📊 Defender Position Distribution:")
    print(defenders['defender_position'].value_counts())
    
    # TOP BY EACH POSITION
    positions_to_show = ['CB', 'RB', 'LB', 'RWB', 'LWB']
    
    for pos in positions_to_show:
        pos_players = defenders[defenders['defender_position'] == pos]
        if len(pos_players) > 0:
            print(f"\n🏆 TOP 10 {pos}s (2024):")
            print("-" * 80)
            
            # Sort by appropriate metric
            if 'WB' in pos:
                sort_col = 'attacking_fullback_index'
            elif pos == 'CB':
                sort_col = 'pure_defender_index'
            else:
                sort_col = 'ball_playing_defender_index'
            
            top_pos = pos_players.nlargest(10, sort_col)[
                ['player', 'squad', 'comp', sort_col, 'tackles', 'progressive_passes', 'defender_position']
            ]
            print(top_pos.to_string(index=False))

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n" + "="*80)
print("💾 SAVING ROLE IDENTIFICATION RESULTS")
print("="*80)

# Combine all with role identifications
if len(forwards) > 0:
    forwards_output = forwards[['player', 'squad', 'comp', 'age', 'forward_role', 
                                 'poacher_score', 'false9_score', 'role_strength',
                                 'goals_per_90', 'key_passes', 'assists_per_90']].copy()
    forwards_path = '/mnt/user-data/outputs/FORWARDS_ROLE_CLASSIFICATION.csv'
    forwards_output.to_csv(forwards_path, index=False)
    print(f"✓ Saved forward roles to: {forwards_path}")

if len(midfielders) > 0:
    midfielders_output = midfielders[['player', 'squad', 'comp', 'age', 'midfielder_role',
                                       'cam_score', 'cdm_score', 'b2b_score', 'role_strength',
                                       'key_passes', 'tackles', 'progressive_passes']].copy()
    midfielders_path = '/mnt/user-data/outputs/MIDFIELDERS_ROLE_CLASSIFICATION.csv'
    midfielders_output.to_csv(midfielders_path, index=False)
    print(f"✓ Saved midfielder roles to: {midfielders_path}")

if len(defenders) > 0 and 'defender_position' in defenders.columns:
    defenders_output = defenders[['player', 'squad', 'comp', 'age', 'defender_position',
                                   'pure_defender_index', 'ball_playing_defender_index',
                                   'attacking_fullback_index', 'tracking_back_index',
                                   'tackles', 'progressive_passes']].copy()
    defenders_path = '/mnt/user-data/outputs/DEFENDERS_POSITION_CLASSIFICATION.csv'
    defenders_output.to_csv(defenders_path, index=False)
    print(f"✓ Saved defender positions to: {defenders_path}")

print("\n" + "="*80)
print("🎉 ROLE IDENTIFICATION COMPLETE!")
print("="*80)

print(f"\n✅ You now have:")
print(f"   • Forward roles: Poachers vs False 9s")
print(f"   • Midfielder roles: CAM vs CDM vs Box-to-Box")
print(f"   • Defender positions: CB, RB, LB, RWB, LWB")
print(f"   • Role strength scores (how pure each player is in their role)")
print(f"\n💡 Use this for:")
print(f"   • Position-specific scouting")
print(f"   • Tactical system fit")
print(f"   • Transfer target identification")
print(f"   • Squad building recommendations")
print("="*80)
