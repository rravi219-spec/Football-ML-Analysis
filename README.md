# ⚽ Football Player ML Analysis

**Advanced Machine Learning System for Football Player Evaluation with Novel Defensive Metrics**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Accuracy](https://img.shields.io/badge/Accuracy-99.3%25-brightgreen.svg)](documentation/01_MODEL_COMPARISON_RF_VS_XGBOOST.md)

---

## 🎯 Project Overview

A comprehensive machine learning system that evaluates **1,336 football players** across Europe's top 5 leagues, achieving **99.3% accuracy** through novel "tracking back" metrics. The system classifies players into **13 tactical positions** and identifies **€200M+** in market inefficiencies.

### Key Achievements
- ✅ **99.3% accuracy** in defender evaluation (+32.2% improvement over baseline)
- ✅ **Novel innovation**: Tracking back metrics measuring defensive actions across all pitch thirds
- ✅ **13 tactical positions** classified with position-specific models
- ✅ **€200M+ value** identified across 408 undervalued players
- ✅ **Dual analysis**: Both current season (2024) and career (2018-2024) rankings

---

## 🔥 The Innovation: Tracking Back Metrics

Traditional defensive metrics only measure tackles and clearances in defensive areas. Modern defenders contribute across the entire pitch. Our breakthrough:

```python
# Weighted defensive work rate based on pitch location
defensive_work_rate = (
    (tackles_def_3rd * 1.0) +   # Normal defending
    (tackles_mid_3rd * 1.2) +   # Tracking back from midfield
    (tackles_att_3rd * 1.5) +   # High pressing in attack
    (interceptions * 0.8)        # Reading the game
)
```

**Result:** Defender evaluation accuracy improved from **67.1% → 99.3%** (+32.2%)

---

## 📊 Key Results

### Model Performance
| Position | Random Forest | XGBoost | Improvement |
|----------|--------------|---------|-------------|
| **Defenders** | 67.1% | **99.3%** | +32.2% |
| **Forwards** | 97.2% | **98.8%** | +1.6% |
| **Midfielders** | 88.9% | **90.2%** | +1.3% |

### Business Impact
- **408 bargain players** identified (value gap > +10)
- **€200M+** in market opportunities
- **Serie A** identified as best value league (+19.8 avg gap)
- **Defenders** most undervalued position (+18.0 avg gap)

### Top Bargains
1. **Jon Aramburu** (Real Sociedad) - €5M valuation, +39 value gap
2. **Jhon Durán** (Aston Villa) - €35M valuation, +31 value gap
3. **Joshua Kimmich** (Bayern Munich) - €50M valuation, +26 value gap

---

## 📋 13 Tactical Positions Classified

**Forwards (2):**
- ⚽ Poachers (pure goalscorers)
- 🎨 False 9s (creative strikers)

**Wingers (2):**
- ⬅️ LW (Left Wing)
- ➡️ RW (Right Wing)

**Midfielders (4):**
- 🎨 CAM (Attacking Midfielders)
- ⚙️ CM (Central Midfielders)
- 🛡️ CDM (Defensive Midfielders)
- 💪 Box-to-Box (Complete Midfielders)

**Defenders (4):**
- 🏰 CB (Center Backs)
- ⬅️ LB (Left Backs)
- ➡️ RB (Right Backs)
- 🔥 WB (Wing Backs)

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/rravi219-spec/Football-ML-Analysis.git
cd Football-ML-Analysis

# Install dependencies
pip install -r requirements.txt
```

### Usage

```bash
# Run complete pipeline (~2 minutes)
python scripts/01_data_cleaning.py
python scripts/02_feature_engineering.py
python scripts/03_model_training.py
python scripts/04_player_classification.py
python scripts/05_value_analysis.py
python scripts/06_create_visualizations.py
python scripts/07_defender_lb_rb_classification.py
python scripts/08_aggregate_all_seasons.py
python scripts/09_enhanced_midfielders.py
python scripts/10_enhanced_wingers_lw_rw.py
```

**Or run all at once:**
```bash
bash run_pipeline.sh
```

---

## 📁 Project Structure

```
Football-ML-Analysis/
│
├── scripts/                          # 10 Python scripts
│   ├── 01_data_cleaning.py          # Clean raw data
│   ├── 02_feature_engineering.py     # Create 127 features (includes tracking back!)
│   ├── 03_model_training.py          # Train XGBoost vs Random Forest
│   ├── 04_player_classification.py   # Classify into positions
│   ├── 05_value_analysis.py          # Find bargains
│   ├── 06_create_visualizations.py   # Generate 7 charts
│   ├── 07_defender_lb_rb_classification.py  # Separate defenders L/R
│   ├── 08_aggregate_all_seasons.py   # Career analysis (7 years)
│   ├── 09_enhanced_midfielders.py    # 4 midfielder types
│   └── 10_enhanced_wingers_lw_rw.py  # Separate wingers L/R
│
├── visualizations/                   # 7 professional charts (300 DPI)
│   ├── model_comparison.png
│   ├── role_distribution.png
│   ├── top_players_all_positions.png
│   ├── value_vs_performance.png
│   ├── league_analysis.png
│   ├── feature_importance.png
│   └── tracking_back_impact.png
│
├── documentation/                    # Comprehensive docs
│   ├── 01_MODEL_COMPARISON_RF_VS_XGBOOST.md
│   ├── 02_2024_SEASON_RANKINGS_ALL_POSITIONS.md
│   ├── 03_CAREER_RANKINGS_ALL_POSITIONS.md
│   ├── 04_2024_VS_CAREER_COMPARISON.md
│   ├── 05_COMPLETE_PYTHON_SCRIPTS.md
│   └── 06_PROJECT_MASTER_SUMMARY.md
│
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Git ignore file
└── README.md                         # This file
```

---

## 🎯 Methodology

### 1. Data Collection
- **Source:** FBref.com (official Statsbomb data)
- **Coverage:** 5,901 player-season observations
- **Scope:** Top 5 European leagues (2018-2024)
- **Players:** 1,446 unique players

### 2. Feature Engineering (127 Features)
**Forwards (4 features):**
- Poacher Index, Shot Efficiency, Clinical Finisher Score

**Midfielders (9 features):**
- Final Third Involvement, Creative Output, Progressive Midfielder Score
- CAM/CM/CDM/Box-to-Box classification scores

**Defenders (8 features including innovation):**
- **Tracking Back Metrics** (novel contribution)
- Defensive Work Rate (weighted by pitch location)
- Ball-Playing Defender Index

### 3. Model Training
- **Algorithm:** XGBoost (Extreme Gradient Boosting)
- **Validation:** 5-fold cross-validation
- **Train/Test Split:** 80/20
- **Position-Specific:** Separate models for forwards, midfielders, defenders

### 4. Classification System
Players classified based on:
- Statistical thresholds (e.g., poacher_score > 40)
- Within-squad comparative analysis (for L/R positions)
- Multi-metric scoring systems

---

## 📊 Sample Results

### Career Rankings (2018-2024)

**Top Poachers:**
1. Erling Haaland - 97.6 (1.02 goals/90)
2. Robert Lewandowski - 95.5 (0.94 goals/90)
3. Kylian Mbappé - 94.8 (0.99 goals/90)

**Top Central Midfielders:**
1. Luka Modrić - 83.6 (still elite at 39!)
2. Corentin Tolisso - 79.1
3. Granit Xhaka - 77.3

**Top Defenders:**
1. James Tarkowski - 84.3 (7 seasons)
2. Yunis Abdelhamid - 70.9
3. Aaron Wan-Bissaka - 66.3

[See full rankings →](documentation/03_CAREER_RANKINGS_ALL_POSITIONS.md)

---

## 📈 Visualizations

### Model Comparison
![Model Comparison](visualizations/model_comparison.png)

### Value vs Performance
![Value Analysis](visualizations/value_vs_performance.png)

### Feature Importance
![Features](visualizations/feature_importance.png)

[View all visualizations →](visualizations/)

---

## 💡 Key Insights

### 1. Tracking Back Innovation
- Defenders now contribute across entire pitch
- High pressing requires tracking back
- Spatial weighting captures modern defending

### 2. Position-Specific Modeling
- Different positions need different evaluation criteria
- Separate models outperform unified approach
- Feature importance varies dramatically by position

### 3. Career vs Current Form
- Haaland: Career #1, 2024 #12 (injury impact)
- Schick: Career outside top 10, 2024 #1 (breakout)
- Shows importance of dual analysis

### 4. Market Inefficiencies
- Serie A offers best bargains (+19.8 avg gap)
- Defenders most undervalued (+18.0)
- 408 players identified with value > market price

### 5. Tactical Evolution
- Only 15 pure CAMs exist (tactical shift away from #10s)
- CM most common midfielder (190 players)
- Right wingers more productive than left (right-footed dominance)

---

## 🔧 Technical Stack

**Languages & Libraries:**
- Python 3.8+
- pandas (data manipulation)
- numpy (numerical computing)
- scikit-learn (machine learning)
- XGBoost (gradient boosting)
- matplotlib & seaborn (visualization)

**Data Sources:**
- FBref.com (performance statistics)
- Transfermarkt (market valuations)

**Key Techniques:**
- Feature engineering
- Ensemble learning
- Cross-validation
- Statistical inference

---

## 📚 Documentation

Comprehensive documentation available:
- [Model Comparison](documentation/01_MODEL_COMPARISON_RF_VS_XGBOOST.md)
- [2024 Season Rankings](documentation/02_2024_SEASON_RANKINGS_ALL_POSITIONS.md)
- [Career Rankings](documentation/03_CAREER_RANKINGS_ALL_POSITIONS.md)
- [Season vs Career Analysis](documentation/04_2024_VS_CAREER_COMPARISON.md)
- [Complete Scripts Guide](documentation/05_COMPLETE_PYTHON_SCRIPTS.md)
- [Project Summary](documentation/06_PROJECT_MASTER_SUMMARY.md)

---

## 🎓 Applications

### For Football Clubs:
- Identify undervalued transfer targets
- Scout players efficiently (1,000+ analyzed in seconds)
- Make data-driven recruitment decisions
- Evaluate squad depth by tactical position

### For Agents:
- Demonstrate client value objectively
- Support contract negotiations with quantitative evidence
- Identify market positioning opportunities

### For Analysts:
- Study player trajectories and age curves
- Analyze positional evolution over time
- Research tactical trends in modern football

### For Fantasy Football:
- Identify players in peak form (2024 rankings)
- Find consistent performers (career rankings)
- Make data-driven selection decisions

---

## 🚀 Future Enhancements

- [ ] Add goalkeeper evaluation system
- [ ] Expand to additional leagues (e.g., Championship, Eredivisie)
- [ ] Implement injury prediction models
- [ ] Create tactical fit analysis (team style matching)
- [ ] Build interactive dashboard (Streamlit/Dash)
- [ ] Add real-time updating system
- [ ] Develop API for external access

---

## 📊 Statistics

**Data Scale:**
- 5,901 player-season records
- 1,446 unique players
- 7 seasons (2018-2024)
- 127 engineered features

**Code:**
- 10 Python scripts
- ~2,500 lines of code
- ~2 minutes full pipeline runtime

**Accuracy:**
- 99.3% (defenders)
- 98.8% (forwards)
- 90.2% (midfielders)
- 96.1% average across positions

**Business Value:**
- €200M+ opportunities
- 408 bargains identified
- 13 tactical positions
- Dual analysis framework

---

## 👨‍💻 Author

**Rakshith Ravi**

Machine Learning Engineer specializing in sports analytics

- GitHub: [@rravi219-spec](https://github.com/rravi219-spec)
- LinkedIn: [Connect with me](https://linkedin.com/in/your-profile)
- Email: your.email@example.com

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Data Source:** FBref.com (Statsbomb)
- **Market Data:** Transfermarkt.com
- **Inspiration:** Modern football's tactical evolution
- **Special Thanks:** To the sports analytics community

---

## ⭐ Star This Repository

If you find this project useful, please consider giving it a star! It helps others discover the work.

---

**Built with ⚽ and 🤖 by Rakshith Ravi**

*Making football analytics accessible through machine learning*
