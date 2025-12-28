"""
PREMIER LEAGUE DATA CLEANING PIPELINE
Combines and cleans 6 seasons of player data (2019-2025)
"""

import pandas as pd
import numpy as np
import os

print("="*80)
print("PREMIER LEAGUE DATA CLEANING PIPELINE")
print("="*80)

# Define file paths
data_dir = '/mnt/user-data/uploads'
seasons = ['2019-2020', '2020-2021', '2021-2022', '2022-2023', '2023-2024', '2024-2025']

# Step 1: Load all seasons
print("\n📂 STEP 1: LOADING ALL SEASONS")
print("-" * 80)

all_data = []

for season in seasons:
    file_path = f'{data_dir}/{season}.csv'
    print(f"Loading {season}...", end=" ")
    
    try:
        # Read with proper encoding, skip header row
        df = pd.read_csv(file_path, encoding='latin-1', header=0, skiprows=[0])
        
        # Add season column
        df['Season'] = season
        
        all_data.append(df)
        print(f"✓ {len(df)} players loaded")
        
    except Exception as e:
        print(f"✗ Error: {e}")

# Combine all seasons
print(f"\n🔗 Combining all seasons...")
combined_df = pd.concat(all_data, ignore_index=True)
print(f"✓ Total: {len(combined_df)} player-season records (before cleaning)")

# Remove any rows where 'Player' column contains 'Player' (header rows)
combined_df = combined_df[combined_df['Player'] != 'Player']
print(f"✓ After removing header rows: {len(combined_df)} records")

# Step 2: Initial Data Inspection
print("\n" + "="*80)
print("📊 STEP 2: DATA INSPECTION")
print("-" * 80)

print(f"Total rows: {len(combined_df)}")
print(f"Total columns: {len(combined_df.columns)}")
print(f"Seasons: {combined_df['Season'].unique()}")

# Step 3: Handle 'Min' column (convert comma-separated numbers)
print("\n" + "="*80)
print("🔧 STEP 3: FIXING DATA TYPES")
print("-" * 80)

# Convert 'Min' from string with commas to integer
print("Fixing 'Min' column (removing commas)...", end=" ")
combined_df['Min'] = combined_df['Min'].str.replace(',', '').astype(int)
print("✓")

# Convert all numeric columns to proper types
print("Converting numeric columns to correct types...", end=" ")
numeric_cols = ['MP', 'Starts', '90s', 'Gls', 'Ast', 'G+A', 'G-PK', 'PK', 'PKatt', 
                'CrdY', 'CrdR', 'xG', 'npxG', 'xAG', 'npxG+xAG', 'PrgC', 'PrgP', 'PrgR',
                'Gls.1', 'Ast.1', 'G+A.1', 'G-PK.1', 'G+A-PK', 'xG.1', 'xAG.1', 
                'xG+xAG', 'npxG.1', 'npxG+xAG.1', 'Age', 'Born']

for col in numeric_cols:
    if col in combined_df.columns:
        combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')

print("✓")

# Step 4: Clean Position Data
print("\n" + "="*80)
print("⚽ STEP 4: CLEANING POSITION DATA")
print("-" * 80)

print(f"Original position distribution:")
print(combined_df['Pos'].value_counts())

# Simplify positions to primary position only
def simplify_position(pos):
    """Convert multi-position players to their primary position"""
    if pd.isna(pos):
        return 'Unknown'
    
    # Take first position if multiple
    primary = pos.split(',')[0]
    
    # Standardize
    if primary == 'GK':
        return 'GK'
    elif primary == 'DF':
        return 'DF'
    elif primary == 'MF':
        return 'MF'
    elif primary == 'FW':
        return 'FW'
    else:
        return 'Unknown'

combined_df['Position_Clean'] = combined_df['Pos'].apply(simplify_position)

print(f"\nCleaned position distribution:")
print(combined_df['Position_Clean'].value_counts())

# Step 5: Filter by Minimum Minutes
print("\n" + "="*80)
print("⏱️ STEP 5: FILTERING BY MINIMUM MINUTES")
print("-" * 80)

# Keep only players with at least 450 minutes (5 full matches worth)
min_minutes = 450
print(f"Filtering players with < {min_minutes} minutes...")

before = len(combined_df)
combined_df = combined_df[combined_df['Min'] >= min_minutes]
after = len(combined_df)

print(f"✓ Removed {before - after} players")
print(f"✓ Kept {after} players with sufficient playing time")

# Step 5.5: Keep ONLY players who are in 2024-2025 season
print("\n" + "="*80)
print("🎯 STEP 5.5: KEEPING ONLY CURRENT PREMIER LEAGUE PLAYERS")
print("-" * 80)

# Get list of players in 2024-2025 season
current_season_players = combined_df[combined_df['Season'] == '2024-2025']['Player'].unique()
print(f"Players in 2024-2025 season: {len(current_season_players)}")

# Filter to keep only these players (but across ALL seasons)
before_filter = len(combined_df)
combined_df = combined_df[combined_df['Player'].isin(current_season_players)]
after_filter = len(combined_df)

print(f"✓ Removed {before_filter - after_filter} records (players who left the league)")
print(f"✓ Kept {after_filter} records for current EPL players")
print(f"✓ Historical data preserved for {len(current_season_players)} active players")

# Step 6: Handle Missing Values
print("\n" + "="*80)
print("❓ STEP 6: HANDLING MISSING VALUES")
print("-" * 80)

missing = combined_df.isnull().sum()
missing = missing[missing > 0]

if len(missing) > 0:
    print("Missing values found:")
    print(missing)
    
    # Fill Age and Born with median
    if 'Age' in missing.index:
        median_age = combined_df['Age'].median()
        combined_df['Age'].fillna(median_age, inplace=True)
        print(f"✓ Filled Age with median: {median_age}")
    
    if 'Born' in missing.index:
        median_born = combined_df['Born'].median()
        combined_df['Born'].fillna(median_born, inplace=True)
        print(f"✓ Filled Born with median: {median_born}")
    
    # Fill Nation with 'Unknown'
    if 'Nation' in missing.index:
        combined_df['Nation'].fillna('Unknown', inplace=True)
        print(f"✓ Filled Nation with 'Unknown'")
else:
    print("✓ No missing values!")

# Step 7: Rename Duplicate Columns (Per 90 stats)
print("\n" + "="*80)
print("🏷️ STEP 7: RENAMING COLUMNS")
print("-" * 80)

# Columns ending in .1 are "Per 90 Minutes" stats
rename_dict = {
    'Gls.1': 'Gls_per90',
    'Ast.1': 'Ast_per90',
    'G+A.1': 'G+A_per90',
    'G-PK.1': 'G-PK_per90',
    'G+A-PK': 'G+A-PK_per90',
    'xG.1': 'xG_per90',
    'xAG.1': 'xAG_per90',
    'xG+xAG': 'xG+xAG_per90',
    'npxG.1': 'npxG_per90',
    'npxG+xAG.1': 'npxG+xAG_per90'
}

combined_df.rename(columns=rename_dict, inplace=True)
print("✓ Renamed per-90 statistics columns")

# Step 8: Create Additional Features
print("\n" + "="*80)
print("🎯 STEP 8: CREATING ADDITIONAL FEATURES")
print("-" * 80)

# Minutes per match
combined_df['Min_per_Match'] = combined_df['Min'] / combined_df['MP']
print("✓ Created 'Min_per_Match'")

# Goals + Assists per 90
if 'Gls_per90' in combined_df.columns and 'Ast_per90' in combined_df.columns:
    combined_df['Goal_Contributions_per90'] = combined_df['Gls_per90'] + combined_df['Ast_per90']
    print("✓ Created 'Goal_Contributions_per90'")

# Progressive Actions Total
if all(col in combined_df.columns for col in ['PrgC', 'PrgP', 'PrgR']):
    combined_df['Progressive_Actions'] = combined_df['PrgC'] + combined_df['PrgP'] + combined_df['PrgR']
    combined_df['Progressive_Actions_per90'] = combined_df['Progressive_Actions'] / combined_df['90s']
    print("✓ Created 'Progressive_Actions' and 'Progressive_Actions_per90'")

# Step 9: Remove Irrelevant Columns
print("\n" + "="*80)
print("🗑️ STEP 9: REMOVING UNNECESSARY COLUMNS")
print("-" * 80)

# Remove 'Rk' (rank), 'Matches' (link), and original 'Pos' (we have Position_Clean)
cols_to_drop = ['Rk', 'Matches', 'Pos']
combined_df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
print(f"✓ Dropped columns: {cols_to_drop}")

# Step 10: Final Summary
print("\n" + "="*80)
print("📈 STEP 10: FINAL DATASET SUMMARY")
print("-" * 80)

print(f"\n✅ CLEANED DATASET:")
print(f"   Total players: {len(combined_df)}")
print(f"   Columns: {len(combined_df.columns)}")
print(f"   Seasons: {combined_df['Season'].nunique()}")
print(f"   Unique players: {combined_df['Player'].nunique()}")

print(f"\n📊 Players per Position:")
print(combined_df['Position_Clean'].value_counts())

print(f"\n📊 Players per Season:")
print(combined_df['Season'].value_counts().sort_index())

# Save cleaned data
print("\n" + "="*80)
print("💾 SAVING CLEANED DATA")
print("-" * 80)

output_path = '/mnt/user-data/outputs/premier_league_cleaned_2019_2025.csv'
combined_df.to_csv(output_path, index=False)
print(f"✓ Saved to: {output_path}")

# Also save a summary
print("\n📋 Creating Data Summary...")
summary = {
    'Total Records': len(combined_df),
    'Unique Players': combined_df['Player'].nunique(),
    'Seasons Covered': combined_df['Season'].nunique(),
    'Positions': combined_df['Position_Clean'].value_counts().to_dict(),
    'Columns': combined_df.columns.tolist(),
    'Date Range': f"{seasons[0]} to {seasons[-1]}"
}

# Display summary
print("\n" + "="*80)
print("🎉 DATA CLEANING COMPLETE!")
print("="*80)
print(f"\nYou now have a clean dataset with:")
print(f"   • {len(combined_df)} player records")
print(f"   • {len(combined_df.columns)} features")
print(f"   • 6 seasons of data")
print(f"   • Positions: GK, DF, MF, FW")
print(f"   • Per-90 statistics normalized")
print(f"   • Progressive action metrics")
print("\n✓ Ready for feature engineering and modeling!")
print("="*80)

# Show first few rows
print("\n📋 FIRST 5 ROWS OF CLEANED DATA:")
print(combined_df.head())

print("\n📊 COLUMN LIST:")
for i, col in enumerate(combined_df.columns, 1):
    print(f"{i}. {col}")
