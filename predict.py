import os, sys, json
from flask import Flask, request, jsonify
from flask_cors import CORS # 1. IMPORT CORS
import mysql.connector
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, RandomizedSearchCV
from scipy.stats import randint
import numpy as np

app = Flask(__name__)
CORS(app) # 2. INITIALIZE CORS

# --- Configuration & Helper Functions ---

def get_db_connection():
    # 3. USE YOUR BLUEHOST CREDENTIALS
    return mysql.connector.connect(
        host='prosperengineeringteam.com', # Use your public Bluehost domain
        user='prospfv0_scout_owl',
        password='BlueHawks2025!',
        database='prospfv0_frc_scouting'
    )

def load_historical_data(event_name):
    conn = get_db_connection()
    query = """
        SELECT match_no, robot, alliance, points, action, location, result 
        FROM scouting_submissions
        WHERE event_name = %s
    """
    data = pd.read_sql(query, conn, params=(event_name,))
    conn.close()
    data.columns = data.columns.str.strip()
    return data

    
def get_match_robots(event_name, match_no):
    # This function is now ONLY for real matches
    conn = get_db_connection()
    query = """
        SELECT robot, alliance
        FROM active_event
        WHERE event_name = %s AND match_number = %s
    """
    params = (event_name, match_no)
    df = pd.read_sql(query, conn, params=params)
    conn.close()
    if df.empty:
         raise Exception(f"No robots found for event '{event_name}' and match '{match_no}'. Check active_event table.")
    df["robot"] = df["robot"].astype(str).str.strip()
    return df

def compute_linear_slope_and_next(xvals, yvals):
    if len(xvals) < 2:
        return 0.0, 0.0
    linreg = LinearRegression()
    linreg.fit(np.array(xvals).reshape(-1, 1), yvals)
    slope = linreg.coef_[0]
    intercept = linreg.intercept_
    next_x = max(xvals) + 1
    pred_next = slope * next_x + intercept
    return slope, pred_next

def aggregate_data(data):
    if data.empty:
        # Return an empty DataFrame with the expected columns
        cols = ["robot", "total_points", "avg_points_per_match", "matches", "success_rate", "total_events", 
                "most_common_action", "points_slope", "success_rate_slope", "total_events_slope", "predicted_next_points"]
        return pd.DataFrame(columns=cols)

    # Compute match-level aggregates
    match_points_df = (
        data.groupby(["match_no", "robot"])["points"]
        .sum()
        .reset_index()
        .rename(columns={"points": "match_points"})
    )

    temp_df = (
        data.assign(is_success=lambda df: df["result"] == "success")
        .groupby(["match_no", "robot"])
        .agg(
            match_events=("result", "count"),
            match_successes=("is_success", "sum")
        )
        .reset_index()
    )

    match_level = pd.merge(match_points_df, temp_df, on=["match_no", "robot"], how="outer")
    match_level["match_success_rate"] = (
        match_level["match_successes"] / match_level["match_events"]
    ).fillna(0.0)

    # Summarize overall robot performance
    robot_summary = (
        match_level.groupby("robot")["match_points"]
        .agg(total_points="sum", avg_points_per_match="mean", matches="count")
        .reset_index()
    )

    robot_success_rate = (
        data.groupby("robot")["result"]
        .apply(lambda x: (x == "success").sum() / len(x) if len(x) else 0)
        .reset_index(name="success_rate")
    )

    robot_total_events = (
        data.groupby("robot")["result"]
        .count()
        .reset_index(name="total_events")
    )

    robot_common_action = (
        data.groupby("robot")["action"]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown")
        .reset_index(name="most_common_action")
    )

    robot_performance = pd.merge(robot_summary, robot_success_rate, on="robot", how="outer")
    robot_performance = pd.merge(robot_performance, robot_total_events, on="robot", how="outer")
    robot_performance = pd.merge(robot_performance, robot_common_action, on="robot", how="outer")
    robot_performance.fillna({
        "total_points": 0,
        "avg_points_per_match": 0,
        "matches": 0,
        "success_rate": 0,
        "total_events": 0,
        "most_common_action": "Unknown"
    }, inplace=True)

    # Compute slopes and predicted next points
    slopes_data = []
    for robot_id, g in match_level.groupby("robot"):
        g_sorted = g.sort_values("match_no")
        match_nos = g_sorted["match_no"].values
        points_slope, pred_next_points = compute_linear_slope_and_next(
            match_nos, g_sorted["match_points"].values
        )
        succ_slope, _ = compute_linear_slope_and_next(
            match_nos, g_sorted["match_success_rate"].values
        )
        evts_slope, _ = compute_linear_slope_and_next(
            match_nos, g_sorted["match_events"].values
        )
        slopes_data.append((robot_id, points_slope, succ_slope, evts_slope, pred_next_points))

    slopes_df = pd.DataFrame(slopes_data, columns=[
        "robot", "points_slope", "success_rate_slope",
        "total_events_slope", "predicted_next_points"
    ])

    if not slopes_df.empty:
        robot_performance = pd.merge(robot_performance, slopes_df, on="robot", how="left")
    else:
        for col in ["points_slope", "success_rate_slope", "total_events_slope", "predicted_next_points"]:
            robot_performance[col] = 0.0

    for col in ["points_slope", "success_rate_slope", "total_events_slope", "predicted_next_points"]:
        robot_performance[col] = robot_performance[col].fillna(0.0)
    
    return robot_performance

def train_model(robot_performance):
    X = robot_performance[
        ["robot", "avg_points_per_match", "success_rate",
         "total_events", "points_slope", "success_rate_slope",
         "total_events_slope"]
    ].copy()
    y = robot_performance["total_points"]
    X["robot"] = X["robot"].astype("category").cat.codes
    
    param_distributions = {
        "n_estimators": randint(100, 201),
        "max_depth": [10, 20, 30, None],
        "min_samples_split": randint(2, 11),
        "min_samples_leaf": randint(1, 5),
        "max_features": ["sqrt", "log2", 0.5],
        "bootstrap": [True, False]
    }
    
    cv_strategy = KFold(n_splits=3, shuffle=True, random_state=42)
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_distributions,
        n_iter=20,
        scoring="neg_mean_squared_error",
        cv=cv_strategy,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    random_search.fit(X, y)
    return random_search.best_estimator_

def robot_features_for_model(robot_data):
    df = pd.DataFrame([robot_data], columns=[
        "robot", "avg_points_per_match", "success_rate",
        "total_events", "points_slope", "success_rate_slope",
        "total_events_slope"
    ])
    df["robot"] = df["robot"].astype("category").cat.codes
    return df

def estimate_robot_performance(robot_id, global_avgs):
    return {
        "robot": robot_id,
        "total_points": 0.0,
        "avg_points_per_match": global_avgs["avg_points_per_match"],
        "success_rate": global_avgs["success_rate"],
        "total_events": global_avgs["total_events"],
        "matches": 0,
        "points_slope": global_avgs["points_slope"],
        "success_rate_slope": global_avgs["success_rate_slope"],
        "total_events_slope": global_avgs["total_events_slope"],
        "predicted_next_points": global_avgs["predicted_next_points"],
        "most_common_action": "Unknown"
    }

def balanced_robot_prediction(robot_id, robot_performance, best_rf, hist_weight, global_avgs):
    robot_id = str(robot_id).strip()
    if robot_id not in robot_performance["robot"].values:
        print(f"Robot {robot_id} missing in aggregated data; using defaults.")
        row = estimate_robot_performance(robot_id, global_avgs)
    else:
        row = robot_performance[robot_performance["robot"] == robot_id].iloc[0].to_dict()
    
    matches_played = row["matches"]
    historical_avg = row["avg_points_per_match"]
    if matches_played < 1:
        model_pred_per_match = historical_avg
    else:
        features = robot_features_for_model(row)
        model_pred_total = best_rf.predict(features)[0]
        model_pred_per_match = model_pred_total / matches_played
    combined_score = hist_weight * historical_avg + (1 - hist_weight) * model_pred_per_match
    return combined_score

def predict_alliance_score(alliance, robot_performance, best_rf, hist_weight, global_avgs):
    total_score = 0.0
    contributions = []
    for robot_id in alliance:
        pts = balanced_robot_prediction(robot_id, robot_performance, best_rf, hist_weight, global_avgs)
        total_score += pts
        contributions.append({"robot": robot_id, "predicted_ppm": pts})
    return total_score, contributions

def get_alliance_stats(alliance, robot_performance, global_avgs):
    alliance_data = []
    for robot_id in alliance:
        robot_id_str = str(robot_id).strip()
        if robot_id_str in robot_performance["robot"].values:
            row = robot_performance[robot_performance["robot"] == robot_id_str].iloc[0].to_dict()
        else:
            row = estimate_robot_performance(robot_id_str, global_avgs)
        alliance_data.append(row)
    
    columns_order = [
        "robot", "total_points", "avg_points_per_match", "success_rate",
        "success_rate_slope", "total_events", "total_events_slope",
        "points_slope", "predicted_next_points", "matches", "most_common_action"
    ]
    
    ordered_alliance_data = []
    for item in alliance_data:
        ordered_item = {
            col: (round(item.get(col, 0), 2) if isinstance(item.get(col, 0), float) else item.get(col))
            for col in columns_order
        }
        ordered_alliance_data.append(ordered_item)
    
    return ordered_alliance_data


# --- API Endpoint ---

@app.route('/predict', methods=['GET'])
def predict():
    try:
        event_name = request.args.get('event_name')
        if not event_name:
            return jsonify({"error": "event_name parameter is required"}), 400

        match_no = int(request.args.get('match_no'))
        
        # 4. FIX: Handle custom match (1313) vs. real match
        if match_no == 1313:
            blue_alliance_param = request.args.get('blue_alliance')
            if not blue_alliance_param:
                return jsonify({"error": "blue_alliance parameter is required"}), 400
            blue_alliance = blue_alliance_param.split(',')
            
            red_alliance_param = request.args.get('red_alliance')
            if not red_alliance_param:
                return jsonify({"error": "red_alliance parameter is required"}), 400
            red_alliance = red_alliance_param.split(',')
        else:
            match_robots = get_match_robots(event_name, match_no)
            blue_alliance = match_robots[match_robots["alliance"].str.lower() == "blue"]["robot"].tolist()
            red_alliance = match_robots[match_robots["alliance"].str.lower() == "red"]["robot"].tolist()
        
        print("Blue Alliance:", blue_alliance, file=sys.stderr)
        print("Red Alliance:", red_alliance, file=sys.stderr)
        
        hist_weight = float(request.args.get('hist_weight', 0.5))
        hist_weight = max(0.0, min(1.0, hist_weight))
        
        # NOTE: Your 'aggregate_data' function doesn't use game config,
        # so we don't need to load the game JSON file here.
        # This is simpler and matches your code.

        data = load_historical_data(event_name)
        if data.empty:
             return jsonify({"error": f"No historical data found for event {event_name}"}), 404
             
        robot_perf = aggregate_data(data)
        
        if robot_perf.empty:
            return jsonify({"error": f"No robot performance data could be aggregated for {event_name}"}), 404

        global_avgs = {
            "avg_points_per_match": robot_perf["avg_points_per_match"].mean(),
            "success_rate": robot_perf["success_rate"].mean(),
            "total_events": robot_perf["total_events"].mean(),
            "points_slope": robot_perf["points_slope"].mean(),
            "success_rate_slope": robot_perf["success_rate_slope"].mean(),
            "total_events_slope": robot_perf["total_events_slope"],
            "predicted_next_points": robot_perf["predicted_next_points"].mean()
        }
        # Handle NaN from empty/single-robot event
        global_avgs = {k: (0 if pd.isna(v) else v) for k, v in global_avgs.items()}
        
        best_rf = train_model(robot_perf)
        
        blue_score, blue_contrib = predict_alliance_score(blue_alliance, robot_perf, best_rf, hist_weight, global_avgs)
        red_score, red_contrib = predict_alliance_score(red_alliance, robot_perf, best_rf, hist_weight, global_avgs)
        blue_stats = get_alliance_stats(blue_alliance, robot_perf, global_avgs)
        red_stats = get_alliance_stats(red_alliance, robot_perf, global_avgs)
        
        if blue_score > red_score:
            winner = "Blue Alliance"
        elif red_score > blue_score:
            winner = "Red Alliance"
        else:
            winner = "Tie"
        
        result = {
            "event_name": event_name,
            "match_no": match_no,
            "blue_score": round(blue_score, 2),
            "blue_contributions": blue_contrib,
            "blue_stats": blue_stats,
            "red_score": round(red_score, 2),
            "red_contributions": red_contrib,
            "red_stats": red_stats,
            "predicted_winner": winner
        }
        return jsonify(result)
    
    except Exception as e:
        # Print the full error to the server logs
        print(f"Exception in /predict: {str(e)}", file=sys.stderr)
        # Return a JSON-formatted error to the browser
        return jsonify({"error": f"Python script failed: {str(e)}"}), 500

# 5. This is the entry point Plesk/Passenger will use
application = app

# Server startup for Render.com and local development
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)