# IPL Win Predictor 🏏

A web application that predicts the real-time probability of a team winning an Indian Premier League (IPL) match based on the current game state. This project demonstrates an end-to-end machine learning workflow, from data analysis and model training to building a web API and deploying a live, interactive frontend.

**Live Application:** [https://ipl-win.netlify.app/]

---

![IPL Win Predictor Screenshot]

## Features

- **Live Win Probability:** Predicts the winning chances for both the batting and bowling teams during the second innings of a match.
- **Dynamic Visualization:** Displays the probabilities on clean, intuitive progress bars that update with each prediction.
- **Match Progress Chart:** A dynamic line chart visualizes how the win probability has shifted throughout the match based on user inputs.
- **State Management:** A "New Match" button allows users to reset the application state and start a new prediction session.
- **Robust Model:** Utilizes a fine-tuned XGBoost Classifier model for high-accuracy predictions.

## Tech Stack

| Category          | Technology                                       |
|-------------------|--------------------------------------------------|
| **Frontend** | HTML, CSS, JavaScript, Tailwind CSS, Chart.js    |
| **Backend** | Python, Flask, Gunicorn                          |
| **Machine Learning**| Scikit-learn, Pandas, NumPy, XGBoost, GridSearchCV |
| **Deployment** | Render (Backend), Netlify (Frontend), Git & GitHub |

## How It Works

The project is built on a decoupled frontend-backend architecture.

#### 1. Data Processing & Model Training (`train_model.py`)
- The model is trained on a historical IPL dataset (`matches.csv` and `deliveries.csv`) containing ball-by-ball data.
- **Feature Engineering:** Key features are engineered from the raw data, including `runs_left`, `balls_left`, `wickets_left`, `current_run_rate`, and `required_run_rate`.
- **Hyperparameter Tuning:** Scikit-learn's `GridSearchCV` is used to systematically find the optimal hyperparameters for the XGBoost model, maximizing its predictive accuracy.
- The trained machine learning pipeline, including the one-hot encoder and the final model, is saved as a `pipe.pkl` file.

#### 2. Backend API (`app.py`)
- A lightweight Flask API is used to serve the trained model.
- The API exposes a `/predict` endpoint that accepts the current match state (teams, city, target, score, etc.) in JSON format.
- It loads the `pipe.pkl` file, processes the input data, and returns the win probabilities for both teams.
- The API is served using a production-ready Gunicorn server and is enabled with CORS to allow requests from the frontend.

#### 3. Frontend (`index.html`)
- A clean, responsive user interface built with HTML and styled with Tailwind CSS.
- JavaScript is used to capture user input and make `fetch` requests to the live backend API.
- The returned probabilities are used to dynamically update the UI elements, including the progress bars and the Chart.js line chart.

## How to Run Locally

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/ipl_win_predictor.git](https://github.com/your-username/ipl_win_predictor.git)
    cd ipl_win_predictor
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3.  **Train the model:**
    Run the training script to generate the `pipe.pkl` file.
    ```bash
    python train_model.py
    ```

4.  **Run the Flask server:**
    ```bash
    python app.py
    ```
    The backend will now be running at `http://127.0.0.1:5000`.

5.  **Open the frontend:**
    Open the `index.html` file in your web browser to use the application.

## Deployment

- The **Flask backend** is deployed as a Web Service on **Render**. The server is run using the `gunicorn app:app` command.
- The **static frontend** is deployed on **Netlify**, which is connected directly to the GitHub repository for continuous deployment.

**Backend URL:** [**Your Backend URL Here**]
