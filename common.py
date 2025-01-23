import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import gym

def load_data(filepath):
    """
    Load and preprocess data from a CSV file.
    """
    data = pd.read_csv(filepath, parse_dates=['Timestamp'], dayfirst=True)
    data['Timestamp'] = pd.to_datetime(data['Timestamp'], format='%d-%m-%Y %H:%M')
    data.set_index('Timestamp', inplace=True)

    scaler = MinMaxScaler()
    scaled_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD', 
                      'MACD_signal', 'MACD_hist', 'SMA_50', 'SMA_200', 'EMA_50', 'EMA_200']
    data[scaled_columns] = scaler.fit_transform(data[scaled_columns])
    return data

class CryptoTradingEnv(gym.Env):
    """
    Custom Gym environment for cryptocurrency trading.
    """
    def __init__(self, data, ensemble_models, window_size=10):
        super(CryptoTradingEnv, self).__init__()
        self.data = data
        self.ensemble_models = ensemble_models
        self.window_size = window_size
        self.current_step = 0
        self.action_space = gym.spaces.Discrete(3)  # Buy, Hold, Sell
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(self.window_size, data.shape[1] + 1), dtype=np.float32  # +1 for ML prediction
        )

    def reset(self):
        self.current_step = self.window_size
        return self._next_observation()

    def _next_observation(self):
        obs = np.array(self.data.iloc[self.current_step - self.window_size:self.current_step])
        features = self.data.iloc[self.current_step][['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD', 'SMA_50', 'SMA_200']].values
        prediction, confidence = self._ensemble_prediction(features)
        return np.hstack([obs, np.full((self.window_size, 1), prediction)])  # Add prediction as a feature

    def _ensemble_prediction(self, features):
        # Convert the features array into a DataFrame with valid feature names
        feature_names = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD', 'SMA_50', 'SMA_200']
        features_df = pd.DataFrame([features], columns=feature_names)
        
        predictions = [model.predict(features_df)[0] for model in self.ensemble_models]
        confidences = [max(model.predict_proba(features_df)[0]) for model in self.ensemble_models]
        
        avg_prediction = round(sum(predictions) / len(predictions))  # Majority vote
        avg_confidence = sum(confidences) / len(confidences)         # Average confidence
        return avg_prediction, avg_confidence


    def step(self, action):
        prev_state = self.data.iloc[self.current_step - 1]
        current_state = self.data.iloc[self.current_step]
        features = current_state[['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD', 'SMA_50', 'SMA_200']].values
        prediction, confidence = self._ensemble_prediction(features)

        reward = 0
        if action == 1:  # Buy
            reward = current_state['Close'] - prev_state['Close']
        elif action == 2:  # Sell
            reward = prev_state['Close'] - current_state['Close']

        if (action == 1 and prediction == 1) or (action == 2 and prediction == 0):
            reward *= (1 + confidence)
        else:
            reward *= (1 - confidence)

        self.current_step += 1
        done = self.current_step >= len(self.data)
        return self._next_observation(), reward, done, {}
