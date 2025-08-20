import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import pickle


def train_and_save_model(matches_path='matches.csv', deliveries_path='deliveries.csv', model_output_path='pipe.pkl'):
    """
    Loads data, preprocesses it, uses GridSearchCV to find the best hyperparameters
    for an XGBoost model, trains it, and saves the best pipeline to a file.
    The model is configured to handle unknown categories gracefully.
    """
    print("--- Starting Model Training with GridSearchCV ---")

    # --- 1. Data Loading and Preprocessing ---
    print("Step 1/5: Loading and preprocessing data...")
    try:
        match = pd.read_csv(matches_path)
        delivery = pd.read_csv(deliveries_path)
    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure '{matches_path}' and '{deliveries_path}' are in the correct directory.")
        return

    total_score_df = delivery.groupby(['match_id', 'inning']).sum()['total_runs'].reset_index()
    total_score_df['total_runs'] = total_score_df['total_runs'] + 1
    match_df = match.merge(total_score_df[['match_id', 'total_runs']], left_on='id', right_on='match_id')

    # --- 2. Data Cleaning and Transformation ---
    print("Step 2/5: Cleaning and transforming data...")
    match_df['team1'] = match_df['team1'].str.replace('Delhi Daredevils', 'Delhi Capitals')
    match_df['team2'] = match_df['team2'].str.replace('Delhi Daredevils', 'Delhi Capitals')
    match_df['team1'] = match_df['team1'].str.replace('Deccan Chargers', 'Sunrisers Hyderabad')
    match_df['team2'] = match_df['team2'].str.replace('Deccan Chargers', 'Sunrisers Hyderabad')

    teams = [
        'Sunrisers Hyderabad', 'Mumbai Indians', 'Gujarat Titans',
        'Royal Challengers Bengaluru', 'Kolkata Knight Riders',
        'Punjab Kings', 'Chennai Super Kings', 'Rajasthan Royals',
        'Delhi Capitals', 'Lucknow Super Giants'
    ]
    match_df = match_df[match_df['team1'].isin(teams)]
    match_df = match_df[match_df['team2'].isin(teams)]

    delivery_df = match_df.merge(delivery, on='match_id')
    delivery_df = delivery_df[delivery_df['inning'] == 2]

    # --- 3. Feature Engineering ---
    print("Step 3/5: Engineering features...")
    delivery_df['current_score'] = delivery_df.groupby('match_id')['total_runs_y'].cumsum()

    # FIX: The line to create 'runs_left' was missing and has been added back.
    delivery_df['runs_left'] = delivery_df['total_runs_x'] - delivery_df['current_score']

    delivery_df.loc[delivery_df['runs_left'] < 0, 'runs_left'] = 0
    delivery_df['balls_left'] = 120 - (delivery_df['over'] * 6 + delivery_df['ball'])
    delivery_df['player_dismissed'] = delivery_df['player_dismissed'].fillna("0").apply(
        lambda x: "1" if x != "0" else "0").astype('int')
    wickets = delivery_df.groupby('match_id')['player_dismissed'].cumsum().values
    delivery_df['wickets_left'] = 10 - wickets
    delivery_df['crr'] = (delivery_df['current_score'] * 6) / (120 - delivery_df['balls_left'])
    delivery_df['rrr'] = (delivery_df['runs_left'] * 6) / delivery_df['balls_left']
    delivery_df['result'] = delivery_df.apply(lambda row: 1 if row['batting_team'] == row['winner'] else 0, axis=1)

    # --- 4. Final DataFrame for Model Training ---
    print("Step 4/5: Finalizing data for training...")
    final_df = delivery_df[[
        'batting_team', 'bowling_team', 'city', 'runs_left', 'balls_left',
        'wickets_left', 'total_runs_x', 'crr', 'rrr', 'result'
    ]]
    final_df = final_df.dropna()
    final_df = final_df[final_df['balls_left'] != 0]

    # --- 5. Model Building and Hyperparameter Tuning ---
    print("Step 5/5: Building model and tuning with GridSearchCV...")
    X = final_df.iloc[:, :-1]
    y = final_df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    # THE FIX IS HERE: Added handle_unknown='ignore' to the OneHotEncoder
    trf = ColumnTransformer([
        ('trf', OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore'),
         ['batting_team', 'bowling_team', 'city'])
    ], remainder='passthrough')

    pipe = Pipeline(steps=[
        ('step1', trf),
        ('step2', XGBClassifier(random_state=1, eval_metric='logloss'))
    ])

    # Define the parameter grid to search
    param_grid = {
        'step2__n_estimators': [100, 200],
        'step2__learning_rate': [0.1, 0.2],
        'step2__max_depth': [5, 7]
    }

    # Initialize GridSearchCV
    grid_search = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1, verbose=2)

    # Fit GridSearchCV to find the best model
    grid_search.fit(X_train, y_train)

    # Get the best estimator
    best_pipe = grid_search.best_estimator_

    print(f"\nBest parameters found: {grid_search.best_params_}")

    y_pred = best_pipe.predict(X_test)
    print(f"Model Accuracy with best parameters: {accuracy_score(y_test, y_pred):.4f}")

    with open(model_output_path, 'wb') as f:
        pickle.dump(best_pipe, f)

    print(f"--- Model training complete. Best pipeline saved to '{model_output_path}' ---")


if __name__ == '__main__':
    train_and_save_model()
