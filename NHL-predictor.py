import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import brier_score_loss, accuracy_score, confusion_matrix, classification_report, roc_auc_score, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
import xgboost as xgb
from datetime import date
import requests
import itertools
from sklearn.calibration import calibration_curve
from sklearn.model_selection import StratifiedKFold
from imblearn.under_sampling import RandomUnderSampler
import numpy as np


def download_file():

    # URL of the CSV file
    url = "https://moneypuck.com/moneypuck/playerData/careers/gameByGame/all_teams.csv"

    # Path where the file will be saved
    save_path = "all_teams.csv"  # You can specify a full path if you want to save it elsewhere

    # Send a GET request to the URL
    response = requests.get(url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Write the content to a local file
        with open(save_path, 'wb') as file:
            file.write(response.content)
        print(f"File downloaded successfully and saved to {save_path}")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")

def tune_xgboost_model(data, use_undersampling=False):
    # Prepare the data
    data = data.dropna()
    X = data.drop(columns=['win'])
    y = data['win']

    # Optional: Undersample the majority class
    if use_undersampling:
        rus = RandomUnderSampler(random_state=42)
        X, y = rus.fit_resample(X, y)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Define a pipeline with scaler and model
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('xgb', xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=42
        ))
    ])

    # Parameter grid for tuning
    param_grid = {
        'xgb__n_estimators': [100, 200],
        'xgb__max_depth': [3, 5, 7],
        'xgb__learning_rate': [0.01, 0.05, 0.1],
        'xgb__subsample': [0.8, 1],
        'xgb__colsample_bytree': [0.8, 1],
        'xgb__reg_lambda': [1, 5],
        'xgb__scale_pos_weight': [1, sum(y == 0) / sum(y == 1)]
    }

    # Cross-validation strategy
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # GridSearchCV setup
    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring='roc_auc',
        verbose=2,
        n_jobs=-1
    )

    print("Starting Grid Search...")
    grid_search.fit(X_train, y_train)
    print("Grid Search Complete.")

    print("Best Parameters:", grid_search.best_params_)
    print("Best AUC Score (CV):", grid_search.best_score_)

    # Evaluate on test set
    best_model = grid_search.best_estimator_
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_pred_proba)
    print("Test AUC:", test_auc)

    return best_model

def tune_random_forest(data):
    # Clean data
    data = data.fillna(data.mean()).dropna()
    X = data.drop(columns=['win'])
    y = data['win']

    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
    )

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Parameter grid
    param_grid = {
        'n_estimators': [200, 300],
        'max_depth': [None, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    param_combos = list(itertools.product(
        param_grid['n_estimators'],
        param_grid['max_depth'],
        param_grid['min_samples_split'],
        param_grid['min_samples_leaf']
    ))

    best_auc = 0
    best_model = None
    best_params = {}

    print(f"Evaluating {len(param_combos)} combinations...")

    for i, (n, d, split, leaf) in enumerate(param_combos, 1):
        print(f"[{i}/{len(param_combos)}] Trying: n_estimators={n}, max_depth={d}, min_samples_split={split}, min_samples_leaf={leaf}")
        
        model = RandomForestClassifier(
            n_estimators=n,
            max_depth=d,
            min_samples_split=split,
            min_samples_leaf=leaf,
            n_jobs=-1,
            random_state=42
        )
        model.fit(X_train_scaled, y_train)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        auc = roc_auc_score(y_test, y_prob)

        print(f"AUC: {auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            best_model = model
            best_params = {
                'n_estimators': n,
                'max_depth': d,
                'min_samples_split': split,
                'min_samples_leaf': leaf
            }

    print("\nBest Parameters:", best_params)
    print("Best AUC Score:", best_auc)

    return best_model, scaler

def tune_logistic_regression(data):
    # Clean data
    data = data.fillna(data.mean()).dropna()
    X = data.drop(columns=['win'])
    y = data['win']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Parameter grid
    penalties = ['l1', 'l2', 'elasticnet']
    Cs = [0.01, 0.1, 0.5, 1, 5, 10]
    solvers = ['liblinear', 'saga']
    l1_ratios = [0.25, 0.5, 0.75]

    best_auc = 0
    best_model = None
    best_params = {}

    print("Starting parameter sweep...")

    for penalty, C, solver in itertools.product(penalties, Cs, solvers):
        # Skip invalid combinations
        if penalty == 'l1' and solver not in ['liblinear', 'saga']:
            continue
        if penalty == 'elasticnet' and solver != 'saga':
            continue
        if penalty == 'l2' and solver not in ['liblinear', 'saga']:
            continue

        l1_ratio_values = l1_ratios if penalty == 'elasticnet' else [None]

        for l1_ratio in l1_ratio_values:
            try:
                model = LogisticRegression(
                    penalty=penalty,
                    C=C,
                    solver=solver,
                    l1_ratio=l1_ratio,
                    max_iter=500,
                    n_jobs=-1,
                    class_weight='balanced',
                    random_state=42
                )
                model.fit(X_train_scaled, y_train)
                y_prob = model.predict_proba(X_test_scaled)[:, 1]
                auc = roc_auc_score(y_test, y_prob)

                print(f"penalty={penalty}, C={C}, solver={solver}, l1_ratio={l1_ratio} => AUC: {auc:.4f}")

                if auc > best_auc:
                    best_auc = auc
                    best_model = model
                    best_params = {
                        'penalty': penalty,
                        'C': C,
                        'solver': solver,
                        'l1_ratio': l1_ratio
                    }

            except Exception as e:
                print(f"Skipped: penalty={penalty}, C={C}, solver={solver}, l1_ratio={l1_ratio} due to error: {e}")

    print("\n✅ Best Parameters:", best_params)
    print(f"🏆 Best AUC Score: {best_auc:.4f}")

    return best_model, scaler

def tune_stacking_final_estimator(X_train_scaled, y_train, xgb_model, rf_model, logreg_model):
   
    # Define base stacked model
    stacked_model = StackingClassifier(
        estimators=[
            ('xgb', xgb_model),
            ('rf', rf_model),
            ('logreg', logreg_model)
        ],
        final_estimator=LogisticRegression(),
        passthrough=False,
        n_jobs=-1
    )

    # Define tuning grid for final estimator
    param_grid = {
        'final_estimator__C': [0.01, 0.1, 1, 10],
        'final_estimator__penalty': ['l2'],
        'final_estimator__solver': ['liblinear']
    }

    # Stratified cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Grid Search
    grid = GridSearchCV(
        estimator=stacked_model,
        param_grid=param_grid,
        cv=cv,
        scoring='roc_auc',
        verbose=2,
        n_jobs=-1
    )

    print("Starting Grid Search for final estimator...")
    grid.fit(X_train_scaled, y_train)
    print("Grid Search Complete.")

    print("Best Parameters:", grid.best_params_)
    print("Best AUC Score:", grid.best_score_)

    return

def evaluate_weight_sets(X_test, y_test, estimators, voting='soft'):

    # Define weight combinations to test
    weight_sets = [
        [4, 2, 1, 1],
        [5, 2, 1, 0],
        [3, 3, 2, 1],
        [4, 3, 1, 1],
        [5, 1, 1, 1]
    ]

    for weights in weight_sets:
        print(f"\nTesting weights: {weights}")
        voting_clf = VotingClassifier(
            estimators=estimators,
            voting=voting,
            weights=weights,
            n_jobs=-1
        )
        
        # Fit the voting classifier on the test set (or use pre-fit models if needed)
        voting_clf.fit(X_test, y_test)  # Or refit using training data if available
        
        y_pred = voting_clf.predict(X_test)
        y_prob = voting_clf.predict_proba(X_test)[:, 1]

        brier = brier_score_loss(y_test, y_prob)
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        logloss = log_loss(y_test, y_prob)

        print(f"Brier Score: {brier:.6f}")
        print(f"Accuracy: {accuracy:.6f}")
        print(f"AUC: {auc:.6f}")
        print(f"Log Loss: {logloss:.6f}")

# Function to calculate rolling averages
def rolling_avg(group, headers, window=32):
    group = group.sort_values('gameDate')
    for header in headers:
        group[f'{header}Avg'] = group[header].shift(1).rolling(window, min_periods=1).mean()
    return group

def get_goalie_data():

    # Define the URL and parameters
    url = "https://api.nhle.com/stats/rest/en/goalie/summary"
    # Define the parameters for the GET request
    params = {
        "isAggregate": "false", # No aggregation
        "isGame": "true", # Stats for individual games
        #Filter for 2023-2024 through 2024-2025 season 
        "cayenneExp": "seasonId>=20222023 and seasonId<=20242025", # Filter for the 2023-2024 season
        "limit": "-1", # No limit, return all results
        'sort':'gameDate',
    }

    # Send the GET request
    response = requests.get(url, params=params)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()
    else:
        print(f"Error: {response.status_code}")

    # Assuming the JSON data contains a list of teams in a key like 'data' or similar
    goalie_data = data.get('data', [])

    # Convert to DataFrame
    goalie_data = pd.DataFrame(goalie_data)
    
    # Define the URL and parameters
    url = "https://api.nhle.com/stats/rest/en/goalie/summary"
    # Define the parameters for the GET request
    params = {
        "isAggregate": "false", # No aggregation
        "isGame": "true", # Stats for individual games
        #Filter for 2023-2024 through 2024-2025 season 
        "cayenneExp": "seasonId>=20192020 and seasonId<=20212022", # Filter for the 2023-2024 season
        "limit": "-1", # No limit, return all results
        'sort':'gameDate',
    }

    # Send the GET request
    response = requests.get(url, params=params)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()
    else:
        print(f"Error: {response.status_code}")

    # Assuming the JSON data contains a list of teams in a key like 'data' or similar
    goalie_data2 = data.get('data', [])
    # Convert to DataFrame
    goalie_data2 = pd.DataFrame(goalie_data2)

     # Merge Goalie Dataframes
    goalie_data = pd.concat([goalie_data, goalie_data2], ignore_index=True)
    
    # Rename columns for consistency
    goalie_data = goalie_data.rename(columns={'teamAbbrev': 'team',
                                              'shotsAgainst': 'shotsGoalie',
                                              'goalsAgainst': 'goalsGoalie',})
    # Seperate revelant columns
    goalie_data = goalie_data[['gameId',
                               'gameDate',
                               'team',
                               'shotsGoalie',
                               'goalsGoalie',
                               'savePct',
                               'saves']]
    
    # Format GameDate
    goalie_data['gameDate'] = pd.to_datetime(goalie_data['gameDate'], format='%Y-%m-%d')

    
    # Keep only the goalie per team per gameDate with the most shots faced
    goalie_data = goalie_data.sort_values(by='shotsGoalie', ascending=False)
    goalie_data = goalie_data.drop_duplicates(subset=['team', 'gameDate'], keep='first')

    goalie_data = goalie_data.sort_values(by='gameDate', ascending=False)

    return goalie_data

# Function to calculate implied probabilities and expected value (EV)
def calculate_equity(row):
    # Convert inputs to numeric safely
    row['home_odds'] = pd.to_numeric(row['home_odds'], errors='coerce')
    row['away_odds'] = pd.to_numeric(row['away_odds'], errors='coerce')
    row['prediction'] = pd.to_numeric(row['prediction'], errors='coerce')

    # Skip if any required values are missing
    if pd.isnull(row['home_odds']) or pd.isnull(row['away_odds']) or pd.isnull(row['prediction']):
        return 0, 0

    # Convert model prediction to win probability
    home_win_prob = row['prediction'] / 100
    away_win_prob = 1 - home_win_prob

    # Calculate home payout on $100 bet
    if row['home_odds'] > 0:
        home_payout = row['home_odds']
    else:
        home_payout = 100 * (100 / abs(row['home_odds']))

    # Calculate away payout on $100 bet
    if row['away_odds'] > 0:
        away_payout = row['away_odds']
    else:
        away_payout = 100 * (100 / abs(row['away_odds']))

    # Calculate expected value (EV)
    home_equity = (home_win_prob * home_payout) - (1 - home_win_prob) * 100
    away_equity = (away_win_prob * away_payout) - (1 - away_win_prob) * 100

    # Don't zero out negative equity — let user decide
    return round(home_equity, 2), round(away_equity, 2)

def calculate_edge(row):
    
    row['home_odds'] = pd.to_numeric(row['home_odds'], errors='coerce')
    row['prediction'] = pd.to_numeric(row['prediction'], errors='coerce')
    
    american_odds = row['home_odds']
    model_prob = row['prediction'] / 100

    if american_odds > 0:
        decimal_odds = (american_odds / 100) + 1
    else:
        decimal_odds = (100 / abs(american_odds)) + 1

    implied_prob = 1 / decimal_odds
    edge = (model_prob - implied_prob) * 100

    return round(implied_prob*100,2)

def generate_match_features(rolling_avg_df, player_team, opposing_team, game_date, home_or_away, season, playoffGame):
    game_date = pd.to_datetime(game_date)

    # Get the latest stats **before** the game date
    player_stats = (
        rolling_avg_df[(rolling_avg_df['team'] == player_team) & (rolling_avg_df['gameDate'] < game_date)]
        .sort_values('gameDate')
        .tail(1)
    )

    opponent_stats = (
        rolling_avg_df[(rolling_avg_df['team'] == opposing_team) & (rolling_avg_df['gameDate'] < game_date)]
        .sort_values('gameDate')
        .tail(1)
    )

    if player_stats.empty or opponent_stats.empty:
        raise ValueError("Insufficient historical data to generate features.")

    # Reset index
    player_stats = player_stats.reset_index(drop=True)
    opponent_stats = opponent_stats.reset_index(drop=True)

    # Add game meta info
    row = {
        'gameDate': game_date,
        'playerTeam': player_team,
        'opposingTeam': opposing_team,
        f'status_{home_or_away.upper()}': 1,
        'playoffGame': playoffGame,
        'season': season
    }

    # Fill in missing status
    if home_or_away.upper() == 'AWAY':
        row['status_HOME'] = 0
    else:
        row['status_AWAY'] = 0

    # Combine features
    features = pd.DataFrame([row])

    # --- Skater stats ---
    skater_cols = ['xGoalsAvg', 'shotAttemptsAvg', 'goalsAvg', 'savedShotsOnGoalAvg',
                'savedUnblockedShotAttemptsAvg', 'penaltiesAvg', 'penaltyMinutesAvg',
                'takeawaysAvg', 'giveawaysAvg', 'shotsOnGoalAvg', 'missedShotsAvg','winsAvg']

    # --- Goalie stats ---
    goalie_cols = ['goalsGoalieAvg', 'savePctAvg', 'shotsGoalieAvg', 'savesAvg']

    # Rename and add player stats
    features = pd.concat([features, player_stats[skater_cols + goalie_cols].rename(
        columns=lambda x: x.replace('Avg', 'For'))], axis=1)

    # Rename and add opponent stats
    features = pd.concat([features, opponent_stats[skater_cols + goalie_cols].rename(
        columns=lambda x: x.replace('Avg', 'Against'))], axis=1)
    
    df = features

    # --- Offensive Stats ---
    df['CF%'] = df['xGoalsFor'] / (df['xGoalsFor'] + df['xGoalsAgainst'])
    df['xGoalsDiff'] = df['xGoalsFor'] - df['xGoalsAgainst']
    df['pointsDiff'] = df['goalsFor'] - df['goalsAgainst']
    df['xShotDiff'] = df['shotsOnGoalFor'] - df['shotsOnGoalAgainst']
    df['xShotAttemptDiff'] = df['shotAttemptsFor'] - df['shotAttemptsAgainst']
    df['FF%'] = (df['shotsOnGoalFor'] + df['missedShotsFor']) - (df['shotsOnGoalAgainst'] + df['missedShotsAgainst'])
   
    # --- Goalie Stats ---
    df['goalieSavePctDiff'] = df['savePctFor'] - df['savePctAgainst']
    df['goalieScoreDiff'] = df['goalsGoalieAgainst'] - df['goalsGoalieFor']  
    df['goalieShotsDiff'] = df['shotsGoalieFor'] - df['shotsGoalieAgainst']
    df['goalieSavesDiff'] = df['savesFor'] - df['savesAgainst']
    
    # ---  Defensive Stats ---
    df['goalieAdjustedSavePctDiff'] = (df['savePctFor'] * df['shotsOnGoalAgainst']) - (df['savePctAgainst'] * df['shotsOnGoalFor'])
    df['goalieExpectedGoalsAgainstDiff'] = (1 - df['savePctFor']) * df['shotsOnGoalAgainst'] - (1 - df['savePctAgainst']) * df['shotsOnGoalFor']
    df['goalieSavesVsShotAttemptsDiff'] = (df['savesFor'] / (df['shotAttemptsAgainst'] + 1e-6)) - (df['savesAgainst'] / (df['shotAttemptsFor'] + 1e-6))
    df['goalieRawStrengthDiff'] = (df['savesFor'] - df['goalsGoalieFor']) - (df['savesAgainst'] - df['goalsGoalieAgainst'])
    df['blockedShotRate'] = df['savedUnblockedShotAttemptsFor'] / (df['shotAttemptsAgainst'] )

    # ---  Discipline and Puck Control ---
    df['penaltyPerTakeawayFor'] = df['penaltiesFor'] / (df['takeawaysFor'] + 1)
    df['penaltyPerTakeawayAgainst'] = df['penaltiesAgainst'] / (df['takeawaysAgainst'] + 1)
    df['controlIndexFor'] = (df['takeawaysFor'] - df['giveawaysFor']) + (df['shotsOnGoalFor'] - df['missedShotsFor'])
    df['controlIndexAgainst'] = (df['takeawaysAgainst'] - df['giveawaysAgainst']) + (df['shotsOnGoalAgainst'] - df['missedShotsAgainst'])

    # ---  Advanced Differential Ratios ---
    df['expectedGoalEfficiency'] = df['xGoalsFor'] / (df['shotAttemptsFor'])
    df['goaliePressureFactor'] = df['shotsGoalieFor'] / (df['shotsGoalieFor'] + df['shotsGoalieAgainst'])
    df['possessionValueRatio'] = df['xGoalsDiff'] / (df['xShotAttemptDiff'] + 1e-6)
    df['xGoalsPerSOG'] = df['xGoalsFor'] / (df['shotsOnGoalFor'] + 1e-6)
    df['goaliePressureRating'] = df['goalieSavePctDiff'] * df['shotsGoalieAgainst']
    df['suppressionIndex'] = df['xGoalsAgainst'] / (df['shotAttemptsAgainst'] + 1e-6)
    df['momentumScore'] = df['xGoalsDiff'] + df['goalieScoreDiff']

    return df

def data_pipeline():

    # Load the data
    file_path = "all_teams.csv"
    df = pd.read_csv(file_path)

    # Scrape goalie stats from NHL API
    goalie_df = get_goalie_data()

    # Drop all situations except 'all'
    df = df[df['situation'] == 'all']
    df = df.drop(columns=['situation', 'team', 'name', 'position'])

    # Ensure the 'gameDate' column is in datetime format
    df['gameDate'] = pd.to_datetime(df['gameDate'], format='%Y%m%d')
    
    # Filter for current season
    df = df[df['gameDate'] >= '2020-10-01']

    df['winsFor'] = df['goalsFor'] > df['goalsAgainst']
    df['winsFor'] = df['winsFor'].astype(int)

    df['winsAgainst'] = df['goalsAgainst'] > df['goalsFor']
    df['winsAgainst'] = df['winsAgainst'].astype(int)

    # Substitue xGoals for flurryAdjustedxGoals
    df = df.drop(columns=['xGoalsFor', 'xGoalsAgainst'])
    df = df.rename(columns={'flurryAdjustedxGoalsFor': 'xGoalsFor', 'flurryAdjustedxGoalsAgainst': 'xGoalsAgainst'})

    # --- Create player data ---
    player_columns = [
        'gameId', 'gameDate', 'playerTeam', 'xGoalsFor', 'shotAttemptsFor', 'goalsFor',
        'savedShotsOnGoalFor', 'savedUnblockedShotAttemptsFor', 'penaltiesFor',
        'penalityMinutesFor', 'takeawaysFor', 'giveawaysFor', 'shotsOnGoalFor', 'missedShotsFor','winsFor']
    player_df = df[player_columns].copy()
    player_df = player_df.rename(columns={
        'playerTeam': 'team',
        'xGoalsFor': 'xGoals',
        'shotAttemptsFor': 'shotAttempts',
        'goalsFor': 'goals',
        'savedShotsOnGoalFor': 'savedShotsOnGoal',
        'savedUnblockedShotAttemptsFor': 'savedUnblockedShotAttempts',
        'penaltiesFor': 'penalties',
        'penalityMinutesFor': 'penaltyMinutes',
        'takeawaysFor': 'takeaways',
        'giveawaysFor': 'giveaways',
        'shotsOnGoalFor': 'shotsOnGoal',
        'missedShotsFor': 'missedShots',
        'winsFor': 'wins',})
    
    player_df['role'] = 'player'

    # --- Create opponent data ---
    opp_columns = [
        'gameId', 'gameDate', 'opposingTeam', 'xGoalsAgainst', 'shotAttemptsAgainst', 'goalsAgainst',
        'savedShotsOnGoalAgainst', 'savedUnblockedShotAttemptsAgainst', 'penaltiesAgainst',
        'penalityMinutesAgainst', 'takeawaysAgainst', 'giveawaysAgainst', 'shotsOnGoalAgainst', 'missedShotsAgainst', 'winsAgainst']
    
    opp_df = df[opp_columns].copy()
    opp_df = opp_df.rename(columns={
        'opposingTeam': 'team',
        'xGoalsAgainst': 'xGoals',
        'shotAttemptsAgainst': 'shotAttempts',
        'goalsAgainst': 'goals',
        'savedShotsOnGoalAgainst': 'savedShotsOnGoal',
        'savedUnblockedShotAttemptsAgainst': 'savedUnblockedShotAttempts',
        'penaltiesAgainst': 'penalties',
        'penalityMinutesAgainst': 'penaltyMinutes',
        'takeawaysAgainst': 'takeaways',
        'giveawaysAgainst': 'giveaways',
        'shotsOnGoalAgainst': 'shotsOnGoal',
        'missedShotsAgainst': 'missedShots',
        'winsAgainst': 'wins',})
    
    opp_df['role'] = 'opponent'

    # Combine player and opponent data
    combined_df = pd.concat([player_df, opp_df], ignore_index=True)
    combined_df.sort_values(by=['team', 'gameDate'], inplace=True)

    # Drop duplicates based on 'team' and 'gameDate' columns
    combined_df = combined_df.drop_duplicates(subset=['team', 'gameDate','role'], keep='last')

    # Compute rolling averages for skater data
    headers = ['xGoals', 'shotAttempts', 'goals', 'savedShotsOnGoal', 'savedUnblockedShotAttempts',
               'penalties', 'penaltyMinutes', 'takeaways', 'giveaways', 'shotsOnGoal', 'missedShots','wins']
    rolling_avg_df = combined_df.groupby('team').apply(rolling_avg, headers=headers).reset_index(drop=True)
    rolling_avg_df.sort_values(by=['team', 'gameDate'], inplace=True)
   

    # --- Prepare game data ---
    df_games = df[['gameDate', 'gameId','playerTeam', 'home_or_away', 'opposingTeam', 'goalsFor', 'goalsAgainst','season','playoffGame']].copy()
    df_games['win'] = df_games['goalsFor'] > df_games['goalsAgainst']
    df_games['win'] = df_games['win'].astype(int)
    df_games.drop(columns=['goalsFor', 'goalsAgainst'], inplace=True)

    # --- Rolling average for goalie data ---
    goalie_headers = ['goalsGoalie', 'savePct', 'shotsGoalie', 'saves']
    goalie_rolling_df = goalie_df.groupby('team').apply(rolling_avg, headers=goalie_headers).reset_index(drop=True)

    # Drop original goalie stats and keep only rolling averages
    goalie_rolling_df.drop(columns=goalie_headers, inplace=True)

    # Combine rolling averages for skaters and goalies
    combined_rolling_df = pd.merge(rolling_avg_df, goalie_rolling_df, 
                                    on=['gameDate', 'team'], 
                                    how='left')

    # --- Merge skater rolling stats with player team ---
    rolling_cols = ['xGoalsAvg', 'shotAttemptsAvg', 'goalsAvg', 'savedShotsOnGoalAvg',
                    'savedUnblockedShotAttemptsAvg', 'penaltiesAvg', 'penaltyMinutesAvg',
                    'takeawaysAvg', 'giveawaysAvg', 'shotsOnGoalAvg', 'missedShotsAvg','winsAvg']

    df_player = pd.merge(df_games, combined_rolling_df, left_on=['gameDate', 'playerTeam'], right_on=['gameDate', 'team'], how='left')

    # Rename skater rolling average columns
    for col in rolling_cols:
        df_player.rename(columns={col: col.replace('Avg', 'For')}, inplace=True)

    # Rename goalie columns to 'For'
    goalie_avg_cols = ['goalsGoalieAvg', 'savePctAvg', 'shotsGoalieAvg', 'savesAvg']
    for col in goalie_avg_cols:
        df_player.rename(columns={col: col.replace('Avg', 'For')}, inplace=True)

    df_player.drop(columns=['team'], inplace=True)

    # --- Merge skater and goalie rolling stats with opponent team ---
    df_opponent = pd.merge(df_games, combined_rolling_df, left_on=['gameDate', 'opposingTeam'], right_on=['gameDate', 'team'], how='left')

    for col in rolling_cols:
        df_opponent.rename(columns={col: col.replace('Avg', 'Against')}, inplace=True)

    for col in goalie_avg_cols:
        df_opponent.rename(columns={col: col.replace('Avg', 'Against')}, inplace=True)

    df_opponent.drop(columns=['team', 'home_or_away'], inplace=True)

    # --- Combine player and opponent stats ---
    player_averages = df_player[['gameDate', 'playerTeam'] + [col.replace('Avg', 'For') for col in rolling_cols] + [col.replace('Avg', 'For') for col in goalie_avg_cols]]
    opponent_averages = df_opponent[['gameDate', 'opposingTeam'] + [col.replace('Avg', 'Against') for col in rolling_cols] + [col.replace('Avg', 'Against') for col in goalie_avg_cols]]

    df_merged = df_games.copy()
    df_merged = pd.merge(df_merged, player_averages, on=['gameDate', 'playerTeam'], how='left')
    df_merged = pd.merge(df_merged, opponent_averages, on=['gameDate', 'opposingTeam'], how='left')
    df_merged = df_merged.drop_duplicates(subset=['gameId','playerTeam'], keep='last')

    # One-hot encode home/away
    df_merged = pd.get_dummies(df_merged, columns=['home_or_away'], prefix=['status'])

    # Rename so stat functions can be swapped with match_features
    df = df_merged
    
    # --- Offensive Stats ---
    df['CF%'] = df['xGoalsFor'] / (df['xGoalsFor'] + df['xGoalsAgainst'])
    df['xGoalsDiff'] = df['xGoalsFor'] - df['xGoalsAgainst']
    df['pointsDiff'] = df['goalsFor'] - df['goalsAgainst']
    df['xShotDiff'] = df['shotsOnGoalFor'] - df['shotsOnGoalAgainst']
    df['xShotAttemptDiff'] = df['shotAttemptsFor'] - df['shotAttemptsAgainst']
    df['FF%'] = (df['shotsOnGoalFor'] + df['missedShotsFor']) - (df['shotsOnGoalAgainst'] + df['missedShotsAgainst'])
   
    # --- Goalie Stats ---
    df['goalieSavePctDiff'] = df['savePctFor'] - df['savePctAgainst']
    df['goalieScoreDiff'] = df['goalsGoalieAgainst'] - df['goalsGoalieFor']  
    df['goalieShotsDiff'] = df['shotsGoalieFor'] - df['shotsGoalieAgainst']
    df['goalieSavesDiff'] = df['savesFor'] - df['savesAgainst']
    
    # ---  Defensive Stats ---
    df['goalieAdjustedSavePctDiff'] = (df['savePctFor'] * df['shotsOnGoalAgainst']) - (df['savePctAgainst'] * df['shotsOnGoalFor'])
    df['goalieExpectedGoalsAgainstDiff'] = (1 - df['savePctFor']) * df['shotsOnGoalAgainst'] - (1 - df['savePctAgainst']) * df['shotsOnGoalFor']
    df['goalieSavesVsShotAttemptsDiff'] = (df['savesFor'] / (df['shotAttemptsAgainst'] + 1e-6)) - (df['savesAgainst'] / (df['shotAttemptsFor'] + 1e-6))
    df['goalieRawStrengthDiff'] = (df['savesFor'] - df['goalsGoalieFor']) - (df['savesAgainst'] - df['goalsGoalieAgainst'])
    df['blockedShotRate'] = df['savedUnblockedShotAttemptsFor'] / (df['shotAttemptsAgainst'] )

    # ---  Discipline and Puck Control ---
    df['penaltyPerTakeawayFor'] = df['penaltiesFor'] / (df['takeawaysFor'] + 1)
    df['penaltyPerTakeawayAgainst'] = df['penaltiesAgainst'] / (df['takeawaysAgainst'] + 1)
    df['controlIndexFor'] = (df['takeawaysFor'] - df['giveawaysFor']) + (df['shotsOnGoalFor'] - df['missedShotsFor'])
    df['controlIndexAgainst'] = (df['takeawaysAgainst'] - df['giveawaysAgainst']) + (df['shotsOnGoalAgainst'] - df['missedShotsAgainst'])

    # ---  Advanced Differential Ratios ---
    df['expectedGoalEfficiency'] = df['xGoalsFor'] / (df['shotAttemptsFor'])
    df['goaliePressureFactor'] = df['shotsGoalieFor'] / (df['shotsGoalieFor'] + df['shotsGoalieAgainst'])
    df['possessionValueRatio'] = df['xGoalsDiff'] / (df['xShotAttemptDiff'] + 1e-6)
    df['xGoalsPerSOG'] = df['xGoalsFor'] / (df['shotsOnGoalFor'] + 1e-6)
    df['goaliePressureRating'] = df['goalieSavePctDiff'] * df['shotsGoalieAgainst']
    df['suppressionIndex'] = df['xGoalsAgainst'] / (df['shotAttemptsAgainst'] + 1e-6)
    df['momentumScore'] = df['xGoalsDiff'] + df['goalieScoreDiff']

    # Encode win and status
    df['status_HOME'] = df['status_HOME'].astype(int)
    df['status_AWAY'] = df['status_AWAY'].astype(int)

    # Save
    df.to_csv('new_standings_output.csv', index=False)
    print("Rolling averages saved to new_standings_output.csv")

    return combined_rolling_df, df

def train_model(data):

    # Fill missing values
    data = data.drop(columns=['gameDate','playerTeam','opposingTeam','gameId'])
    data = data.fillna(data.mean())
    # Shuffle the dataframe
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    # tune_xgboost_model(data)
    # tune_random_forest(data)
    # tune_logistic_regression(data)

    # Features and target
    X = data.drop(columns=['win'])
    y = data['win']

    # Split into training and testing sets with a fixed random state
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)

    rus = RandomUnderSampler(random_state=42)
    X_train, y_train = rus.fit_resample(X_train, y_train)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define XGBoost model
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        n_estimators=500,  # Number of trees
        max_depth=2,  # Depth of the tree
        learning_rate=0.01,  # Step size
        subsample=0.8,  # Fraction of samples to train each tree
        colsample_bytree=1,  # Fraction of features to consider for each tree
        reg_alpha=1,         # L1 regularization (sparse model)
        reg_lambda=2,        # L2 regularization
        gamma=1,  # Regularization term
        n_jobs=-1)

    # Define other models for stacking
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,      
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1)

    logreg_model = LogisticRegression(
        solver='liblinear',
        penalty='l1',
        C=.1,  # slightly more regularization
        random_state=42,
        l1_ratio=None,
        n_jobs=-1)

    # Tune the stacking final estimator
    # tune_stacking_final_estimator(X_train_scaled, y_train, xgb_model, rf_model, logreg_model)

    # Create the Stacking Classifier
    stacked_model = StackingClassifier(
        estimators=[
            ('xgb', xgb_model),
            ('rf', rf_model),
            ('logreg', logreg_model)
        ],
        final_estimator=LogisticRegression(C=1, penalty='l2', n_jobs=-1),
        n_jobs=-1)

    # CalibratedClassifierCV for better probability estimates
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    calibrated_model = CalibratedClassifierCV(estimator=stacked_model, method='isotonic', cv=cv, n_jobs=-1)
    # calibrated_model=stacked_model

    # Train the XGBoost model
    print("Training model ensemble...")
    # tune_xgb_model(X_train_scaled, y_train)
    calibrated_model.fit(X_train_scaled, y_train)

    # Evaluate on test set
    print("Evaluating model on test set...")
    y_pred = calibrated_model.predict(X_test_scaled)
    y_prob = calibrated_model.predict_proba(X_test_scaled)[:, 1] 

    from sklearn.metrics import brier_score_loss
    brier = brier_score_loss(y_test, y_prob)
    print(f'Brier Score: {brier}')

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    logloss = log_loss(y_test, y_prob)
    confusion = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    # Print evaluation
    print(f'Accuracy: {accuracy}')
    print(f'AUC: {auc}')
    print(f'Log Loss: {logloss}')
    # print(f'Confusion Matrix:\n{confusion}')
    # print(f'Classification Report:\n{class_report}')
  
    return calibrated_model, scaler

# Helper function to extract moneyline odds from a list of odds
def extract_moneyline(odds_list):
    for o in odds_list:
        if o.get("providerId") == 8:  # This provider seems to use American odds
            return o["value"]
    return None

def get_schedule():
    today = date.today().strftime("%Y-%m-%d")
    url = f"https://api-web.nhle.com/v1/schedule/{today}"
    response = requests.get(url)

    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        return pd.DataFrame()

    data = response.json()
    game_days = data.get('gameWeek', [])
    df = pd.DataFrame(game_days)
    df = df[df['date'] == today]

    all_games = [game for day in df['games'].tolist() for game in day]
    games_df = pd.DataFrame(all_games)

    game_data = []
    for game in games_df.to_dict('records'):
        game_data.append({
            "game_id": game["id"],
            "start_time_utc": game["startTimeUTC"],
            "home_team": f"{game['homeTeam']['placeName']['default']} {game['homeTeam']['commonName']['default']}",
            "home_abbrev": game["homeTeam"]["abbrev"],
            # "home_odds": extract_moneyline(game["homeTeam"]["odds"]),
            "away_team": f"{game['awayTeam']['placeName']['default']} {game['awayTeam']['commonName']['default']}",
            "away_abbrev": game["awayTeam"]["abbrev"],
            # "away_odds": extract_moneyline(game["awayTeam"]["odds"]),
        })

    # Convert start time to datetime format and drop time
    for game in game_data:
        game["start_time_utc"] = pd.to_datetime(game["start_time_utc"]).date()

    schedule_detailed = pd.DataFrame(game_data)
    # schedule_detailed.to_csv("C:/Users/hd_fr/OneDrive/Python Projects/chell/schedule_detailed.csv")

    # Filter to only include relevant columns
    schedule = schedule_detailed[['home_abbrev', 'away_abbrev', "start_time_utc"]]
    # Rename columns for consistency
    schedule = schedule.rename(columns={
        'home_abbrev': 'playerTeam',
        'away_abbrev': 'opposingTeam',
        'start_time_utc': 'gameDate'
    })

    return schedule, schedule_detailed

def predict_schedule(schedule, schedule_detailed):

    # List to store all feature rows
    all_feature_rows = []

    # Loop through each game in the schedule
    results = []

    for _, row in schedule.iterrows():
        try:
            player_team = row['playerTeam']
            opposing_team = row['opposingTeam']
            game_date = row['gameDate']  # Extract date in YYYY-MM-DD format
            home_or_away = 'HOME'

            # Generate match features
            feature_row = generate_match_features(
                rolling_avg_df,
                player_team=player_team,
                opposing_team=opposing_team,
                game_date=game_date,
                home_or_away=home_or_away,
                season = 2025,
                playoffGame = 0)

            # Append feature row to list
            all_feature_rows.append(feature_row)

            # Drop non-numeric columns from the features
            features_to_scale = feature_row.drop(['gameDate','playerTeam', 'opposingTeam'], axis=1)

            # Ensure all training columns are present
            for col in training_columns:
                if col not in features_to_scale.columns:
                    features_to_scale[col] = 0  # Default value if missing

            # Reorder columns to match training
            features_to_scale = features_to_scale[training_columns]

            # Scale and predict
            scaled_features = scaler.transform(features_to_scale)
            prediction = model.predict_proba(scaled_features.reshape(1, -1))[0][1]
            
            results.append({
                'game_date': game_date,
                'home_team': row['playerTeam'],
                'away_team': row['opposingTeam'],
                'prediction': prediction
            })

        except Exception as e:
            print(f"Error processing game {row['playerTeam']} vs {row['opposingTeam']}: {e}")

    # Create a DataFrame of predictions
    prediction_df = pd.DataFrame(results)
    
    # Save Features as a dataframe
    all_feature_rows = pd.concat(all_feature_rows, ignore_index=True)
    all_feature_rows.to_csv("C:/Users/hd_fr/OneDrive/Python Projects/chell/features.csv", index=False)
    print("Features saved to features.csv")

    # Rename column for merging
    prediction_df.rename(columns={'home_team':'home_abbrev'}, inplace=True)
    prediction_df = prediction_df.drop(columns=['game_date', 'away_team'])

    # Merge with the schedule DataFrame based on home team (home_abbrev)
    final_df = pd.merge(schedule_detailed, prediction_df, on='home_abbrev', how='left')
    # Format the prediction as a percentage
    final_df['prediction'] = (final_df['prediction'] * 100).round(2)
    
    # Calculate Equity
    final_df[['home_equity', 'away_equity']] = final_df.apply(calculate_equity, axis=1, result_type="expand")
    final_df['house_prediction'] = final_df.apply(calculate_edge, axis=1, result_type="expand")

    print(final_df)

    # Save it
    final_df.to_csv("game_predictions.csv", index=False)
    print("Predictions saved to game_predictions.csv")

if __name__ == "__main__":

    # Get and print today's games
    # schedule, schedule_detailed = get_schedule()

    # Download Current Team Data
    download_file()
    # Run the data pipeline to get the rolling averages
    print("Running data pipeline...")
    rolling_avg_df, df = data_pipeline()

    # Train the model
    model, scaler = train_model(df)
    print("Model training complete.")

    # Save the training column order
    df = df.drop(columns=['win','gameId','gameDate','opposingTeam','playerTeam'])
    training_columns = df.columns.tolist()

    # Make new predictions
    # predict_schedule(schedule, schedule_detailed)
 

