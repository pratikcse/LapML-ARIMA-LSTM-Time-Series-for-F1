from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.db.models import Avg
from .models import Race, Driver, LapTime
import requests
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import warnings
from datetime import datetime
import logging
import random
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

def home(request):
    """Render the home page"""
    return render(request, 'race/home.html')

def convert_lap_time_to_seconds(lap_time):
    """Convert lap time format MM:SS.sss to total seconds."""
    try:
        if not lap_time:
            logger.warning("Empty lap time received")
            return None
        
        # Handle different time formats
        if ':' in lap_time:
            minutes, seconds = lap_time.split(':')
            try:
                total_seconds = float(minutes) * 60 + float(seconds)
                logger.info(f"Converted {lap_time} to {total_seconds} seconds")
                return total_seconds
            except ValueError as e:
                logger.error(f"Error converting time parts: {e}")
                return None
        else:
            # If it's already in seconds format
            try:
                seconds = float(lap_time)
                logger.info(f"Converted {lap_time} to {seconds} seconds")
                return seconds
            except ValueError as e:
                logger.error(f"Error converting seconds: {e}")
                return None
    except Exception as e:
        logger.error(f"Error converting lap time '{lap_time}': {e}")
        return None

def calculate_tyre_wear(lap_number):
    """Calculate average tyre wear based on lap number"""
    base_degradation = 0.02  # 2% wear per lap on average
    wear = min(1.0, base_degradation * (lap_number - 1))  # Start from 0% on first lap
    return wear

def calculate_pit_stop_window(lap_times, current_lap):
    """Calculate pit stop window based on lap time degradation"""
    if len(lap_times) < 5:
        return None
    
    # Convert lap times to seconds, filtering out None values
    lap_times_seconds = [lt for lt in [convert_lap_time_to_seconds(lt) for lt in lap_times] if lt is not None]
    
    if len(lap_times_seconds) < 5:
        return None
    
    # Calculate rolling average of last 5 laps
    recent_laps = lap_times_seconds[-5:]
    avg_recent = np.mean(recent_laps)
    baseline = np.mean(lap_times_seconds[:5])  # First 5 laps as baseline
    
    # If recent laps are significantly slower than baseline
    time_difference = avg_recent - baseline
    degradation_threshold = 1.5  # seconds slower than baseline
    
    return {
        'is_in_window': 1.0 if time_difference > degradation_threshold else 0.0,
        'time_difference': float(time_difference)
    }

def predict_next_lap_time(lap_times, current_lap):
    """Predict next lap time using ARIMA"""
    try:
        if len(lap_times) < 3:  # Reduced minimum required laps
            logger.info(f"Not enough lap times for prediction (need 3, got {len(lap_times)})")
            return None, None
        
        # Convert lap times to seconds
        lap_times_seconds = []
        for lt in lap_times:
            seconds = convert_lap_time_to_seconds(lt)
            if seconds is not None:
                lap_times_seconds.append(seconds)
        
        if len(lap_times_seconds) < 3:  # Reduced minimum required laps
            logger.info(f"Not enough valid lap times for prediction (need 3, got {len(lap_times_seconds)})")
            return None, None
        
        # For early laps (less than 5), use simple moving average
        if len(lap_times_seconds) < 5:
            prediction = np.mean(lap_times_seconds)
            logger.info(f"Using simple average for early laps (lap {current_lap})")
            return float(prediction), None
        
        # Apply rolling mean to smooth out noise
        window_size = min(3, len(lap_times_seconds))
        if len(lap_times_seconds) >= window_size:
            lap_times_seconds = np.convolve(lap_times_seconds, np.ones(window_size)/window_size, mode='valid')
        
        # Remove outliers only if we have enough data
        if len(lap_times_seconds) >= 5:
            mean = np.mean(lap_times_seconds)
            std = np.std(lap_times_seconds)
            lap_times_seconds = [x for x in lap_times_seconds if abs(x - mean) <= 2 * std]
        
        if len(lap_times_seconds) < 3:  # Reduced minimum required laps
            logger.info("Not enough lap times after processing")
            return None, None
        
        # Test different ARIMA parameters
        best_aic = float('inf')
        best_model = None
        best_params = None
        
        # Try different ARIMA parameters
        for p in range(2):
            for d in range(2):
                for q in range(2):
                    try:
                        model = ARIMA(lap_times_seconds, order=(p, d, q))
                        model_fit = model.fit()
                        if model_fit.aic < best_aic:
                            best_aic = model_fit.aic
                            best_model = model_fit
                            best_params = (p, d, q)
                    except Exception as e:
                        logger.warning(f"ARIMA model failed for parameters ({p},{d},{q}): {str(e)}")
                        continue
        
        if best_model is None:
            # Fallback to simple moving average if ARIMA fails
            logger.warning("No valid ARIMA model could be fitted, using moving average")
            prediction = np.mean(lap_times_seconds[-3:])  # Use last 3 laps
        else:
            logger.info(f"Best ARIMA parameters: {best_params}")
            prediction = best_model.forecast(steps=1)[0]
        
        # Ensure prediction is within reasonable bounds
        min_lap_time = min(lap_times_seconds) * 0.95
        max_lap_time = max(lap_times_seconds) * 1.10
        prediction = np.clip(prediction, min_lap_time, max_lap_time)
        
        logger.info(f"Lap {current_lap} - Final prediction: {prediction}")
        return float(prediction), None
        
    except Exception as e:
        logger.error(f"Error in predict_next_lap_time: {str(e)}")
        return None, None

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=2, output_size=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 16)
        self.fc2 = nn.Linear(16, output_size)
        
        # Dropout and activation
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))
        
        # Get the last time step output
        out = out[:, -1, :]
        
        # Apply dropout and activation
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

def create_lstm_model(input_size=1, hidden_size=32, num_layers=2, output_size=1):
    """Create and return LSTM model"""
    model = LSTMModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=output_size
    )
    return model

def prepare_lstm_data(lap_times, lookback=5):
    """Prepare data for LSTM model"""
    try:
        # Convert lap times to seconds and filter out None values
        lap_times_seconds = []
        for lt in lap_times:
            seconds = convert_lap_time_to_seconds(lt)
            if seconds is not None:
                lap_times_seconds.append(seconds)
                logger.info(f"Converted lap time {lt} to {seconds} seconds")
        
        if len(lap_times_seconds) < lookback + 1:
            logger.info(f"Not enough data points for LSTM (need {lookback + 1}, got {len(lap_times_seconds)})")
            return None, None, None
        
        # Normalize data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(np.array(lap_times_seconds).reshape(-1, 1))
        
        # Create sequences
        X, y = [], []
        for i in range(len(scaled_data) - lookback):
            X.append(scaled_data[i:(i + lookback), 0])
            y.append(scaled_data[i + lookback, 0])
        
        logger.info(f"Prepared LSTM data: {len(X)} sequences")
        return np.array(X), np.array(y), scaler
        
    except Exception as e:
        logger.error(f"Error in prepare_lstm_data: {str(e)}")
        return None, None, None

def train_lstm_model(X, y, model, criterion, optimizer, num_epochs=50):
    """Train the LSTM model"""
    try:
        X = torch.FloatTensor(X).unsqueeze(-1)
        y = torch.FloatTensor(y).unsqueeze(-1)
        
        # Training loop
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(X)
            loss = criterion(outputs, y)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                logger.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        
        return model
        
    except Exception as e:
        logger.error(f"Error in train_lstm_model: {str(e)}")
        return None

def predict_next_lap_lstm(lap_times, current_lap):
    """Predict next lap time using LSTM"""
    try:
        logger.info(f"Starting LSTM prediction for lap {current_lap} with {len(lap_times)} lap times")
        
        # For first 15 laps, use simple average
        if len(lap_times) < 15:
            # Convert lap times to seconds
            lap_times_seconds = []
            for lt in lap_times:
                seconds = convert_lap_time_to_seconds(lt)
                if seconds is not None:
                    lap_times_seconds.append(seconds)
                    logger.info(f"Converted lap time {lt} to {seconds} seconds")
            
            logger.info(f"Valid lap times for prediction: {len(lap_times_seconds)}")
            
            if len(lap_times_seconds) < 3:
                logger.info(f"Not enough valid lap times for prediction (need 3, got {len(lap_times_seconds)})")
                return None, None
            
            # Use weighted average for early laps (more weight to recent laps)
            weights = np.linspace(0.5, 1.0, len(lap_times_seconds))
            weights = weights / np.sum(weights)
            prediction = np.average(lap_times_seconds, weights=weights)
            
            logger.info(f"Using weighted average for early laps (lap {current_lap}): {prediction}")
            return float(prediction), None
        
        # For laps 15 and beyond, use LSTM
        lookback = min(5, len(lap_times) - 1)  # Adjust lookback based on available data
        X, y, scaler = prepare_lstm_data(lap_times, lookback)
        
        if X is None or len(X) == 0:
            logger.warning("Could not prepare LSTM data, falling back to weighted average")
            lap_times_seconds = [convert_lap_time_to_seconds(lt) for lt in lap_times if convert_lap_time_to_seconds(lt) is not None]
            weights = np.linspace(0.5, 1.0, len(lap_times_seconds))
            weights = weights / np.sum(weights)
            prediction = np.average(lap_times_seconds, weights=weights)
            return float(prediction), None
        
        # Create and train model
        model = create_lstm_model()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Train model
        model = train_lstm_model(X, y, model, criterion, optimizer, num_epochs=50)
        
        # Prepare last sequence for prediction
        last_sequence = torch.FloatTensor(X[-1]).unsqueeze(0).unsqueeze(-1)
        
        # Make prediction
        model.eval()
        with torch.no_grad():
            predicted_scaled = model(last_sequence).numpy()
        
        predicted_time = scaler.inverse_transform(predicted_scaled)[0][0]
        
        # Ensure prediction is within reasonable bounds
        min_lap_time = min([convert_lap_time_to_seconds(lt) for lt in lap_times if convert_lap_time_to_seconds(lt) is not None]) * 0.95
        max_lap_time = max([convert_lap_time_to_seconds(lt) for lt in lap_times if convert_lap_time_to_seconds(lt) is not None]) * 1.10
        predicted_time = np.clip(predicted_time, min_lap_time, max_lap_time)
        
        logger.info(f"LSTM prediction for lap {current_lap}: {predicted_time}")
        return float(predicted_time), None
        
    except Exception as e:
        logger.error(f"Error in LSTM prediction: {str(e)}")
        # Fallback to weighted average if LSTM fails
        try:
            lap_times_seconds = [convert_lap_time_to_seconds(lt) for lt in lap_times if convert_lap_time_to_seconds(lt) is not None]
            if len(lap_times_seconds) >= 3:
                weights = np.linspace(0.5, 1.0, len(lap_times_seconds))
                weights = weights / np.sum(weights)
                prediction = np.average(lap_times_seconds, weights=weights)
                logger.info(f"Using weighted average fallback for lap {current_lap}: {prediction}")
                return float(prediction), None
        except Exception as fallback_error:
            logger.error(f"Fallback prediction also failed: {str(fallback_error)}")
        return None, None

def fetch_race_data(request):
    if request.method == 'POST':
        year = request.POST.get('year', 2023)
        round_number = request.POST.get('round', 1)
        
        # Fetch race data from Ergast API
        base_url = f"http://ergast.com/api/f1/{year}/{round_number}.json"
        response = requests.get(base_url)
        data = response.json()
        
        race_data = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
        if not race_data:
            return JsonResponse({'error': 'No race data found'}, status=404)
        
        race_info = race_data[0]
        race, created = Race.objects.get_or_create(
            year=year,
            round_number=round_number,
            defaults={
                'race_name': race_info.get('raceName', ''),
                'date': datetime.strptime(race_info.get('date', ''), '%Y-%m-%d').date(),
                'circuit_name': race_info.get('Circuit', {}).get('circuitName', '')
            }
        )
        
        # Fetch drivers
        drivers_url = f"http://ergast.com/api/f1/{year}/drivers.json"
        drivers_response = requests.get(drivers_url)
        drivers_data = drivers_response.json()
        
        for driver in drivers_data.get("MRData", {}).get("DriverTable", {}).get("Drivers", []):
            Driver.objects.get_or_create(
                driver_id=driver.get('driverId', ''),
                defaults={
                    'code': driver.get('code', ''),
                    'first_name': driver.get('givenName', ''),
                    'last_name': driver.get('familyName', '')
                }
            )
        
        return JsonResponse({'success': True, 'race_id': race.id})
    
    return JsonResponse({'error': 'Invalid request method'}, status=400)

@csrf_exempt
def race_simulation(request):
    """View for ARIMA-based race simulation"""
    if request.method == 'GET':
        races = Race.objects.all()
        drivers = Driver.objects.all()
        return render(request, 'race/simulation.html', {
            'races': races,
            'drivers': drivers
        })
    
    elif request.method == 'POST':
        try:
            race_id = request.POST.get('race_id')
            driver_id = request.POST.get('driver_id')
            
            if not race_id or not driver_id:
                return JsonResponse({'success': False, 'error': 'Missing race_id or driver_id'})
            
            # Get race and driver
            race = Race.objects.get(id=race_id)
            driver = Driver.objects.get(driver_id=driver_id)
            
            # Get lap times from database
            lap_times = LapTime.objects.filter(
                race=race,
                driver=driver
            ).order_by('lap_number')
            
            # If no lap times in database, fetch from Ergast API
            if not lap_times.exists():
                logger.info(f"No lap times found in database. Fetching from Ergast API...")
                lap_number = 1
                while True:
                    lap_url = f"http://ergast.com/api/f1/{race.year}/{race.round_number}/laps/{lap_number}.json"
                    response = requests.get(lap_url)
                    data = response.json()
                    
                    races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
                    if not races:
                        break
                    
                    laps = races[0].get("Laps", [])
                    if not laps:
                        break
                    
                    for lap in laps:
                        for timing in lap.get("Timings", []):
                            if timing["driverId"] == driver.driver_id:
                                lap_time_seconds = convert_lap_time_to_seconds(timing["time"])
                                if lap_time_seconds is not None:
                                    LapTime.objects.create(
                                        race=race,
                                        driver=driver,
                                        lap_number=int(lap["number"]),
                                        lap_time=timing["time"],
                                        lap_time_seconds=lap_time_seconds
                                    )
                    
                    lap_number += 1
                
                # Refresh lap times after fetching
                lap_times = LapTime.objects.filter(race=race, driver=driver).order_by('lap_number')
            
            # Process lap times
            lap_data = []
            total_laps = len(lap_times)
            
            # Get all lap times for prediction
            all_lap_times = [lt.lap_time for lt in lap_times]
            
            for i, lap_time in enumerate(lap_times):
                lap_number = lap_time.lap_number
                actual_time = lap_time.lap_time
                actual_time_seconds = convert_lap_time_to_seconds(actual_time)
                
                if actual_time_seconds is None:
                    continue
                
                # Predict next lap time using ARIMA
                predicted_time, _ = predict_next_lap_time(
                    all_lap_times[:i+1],
                    lap_number
                )
                
                # Calculate prediction error if we have the next lap time
                prediction_error = None
                if i < total_laps - 1:
                    next_lap_time = convert_lap_time_to_seconds(lap_times[i+1].lap_time)
                    if predicted_time is not None and next_lap_time is not None:
                        prediction_error = abs(predicted_time - next_lap_time)
                
                # Add lap data even if prediction is None
                lap_data.append({
                    'lap_number': lap_number,
                    'lap_time': actual_time,
                    'predicted_next_lap': predicted_time,
                    'prediction_error': prediction_error
                })
                
                # Log prediction status
                logger.info(f"Lap {lap_number}: Actual={actual_time}, Predicted={predicted_time}, Error={prediction_error}")
            
            # Calculate statistics
            valid_lap_times = [lt for lt in lap_times if convert_lap_time_to_seconds(lt.lap_time) is not None]
            if valid_lap_times:
                fastest_lap = min(valid_lap_times, key=lambda x: convert_lap_time_to_seconds(x.lap_time))
                avg_lap_time = np.mean([convert_lap_time_to_seconds(lt.lap_time) for lt in valid_lap_times])
            else:
                fastest_lap = None
                avg_lap_time = 0.0
            
            prediction_errors = [lap.get('prediction_error', 0) for lap in lap_data if 'prediction_error' in lap and lap['prediction_error'] is not None]
            avg_prediction_error = np.mean(prediction_errors) if prediction_errors else None
            
            # Calculate MSE and RMSE
            mse = np.mean([error ** 2 for error in prediction_errors]) if prediction_errors else None
            rmse = np.sqrt(mse) if mse is not None else None
            
            statistics = {
                'total_laps': total_laps,
                'avg_lap_time': float(avg_lap_time),
                'fastest_lap': {
                    'lap_number': fastest_lap.lap_number if fastest_lap else 0,
                    'time': fastest_lap.lap_time if fastest_lap else "N/A"
                },
                'avg_prediction_error': float(avg_prediction_error) if avg_prediction_error is not None else None,
                'mse': float(mse) if mse is not None else None,
                'rmse': float(rmse) if rmse is not None else None
            }
            
            return JsonResponse({
                'success': True,
                'lap_data': lap_data,
                'statistics': statistics
            })
            
        except Exception as e:
            logger.error(f"Error in race_simulation: {str(e)}")
            return JsonResponse({'success': False, 'error': str(e)})
    
    return JsonResponse({'success': False, 'error': 'Invalid request method'})

@csrf_exempt
def lstm_simulation(request):
    """View for LSTM-based race simulation"""
    if request.method == 'GET':
        races = Race.objects.all()
        drivers = Driver.objects.all()
        return render(request, 'race/lstm_simulation.html', {
            'races': races,
            'drivers': drivers
        })
    
    elif request.method == 'POST':
        try:
            race_id = request.POST.get('race_id')
            driver_id = request.POST.get('driver_id')
            
            logger.info(f"Starting LSTM simulation for race_id: {race_id}, driver_id: {driver_id}")
            
            if not race_id or not driver_id:
                return JsonResponse({'success': False, 'error': 'Missing race_id or driver_id'})
            
            # Get race and driver
            race = Race.objects.get(id=race_id)
            driver = Driver.objects.get(driver_id=driver_id)
            
            # Get lap times from database
            lap_times = LapTime.objects.filter(
                race=race,
                driver=driver
            ).order_by('lap_number')
            
            logger.info(f"Found {lap_times.count()} lap times in database")
            
            # If no lap times in database, fetch from Ergast API
            if not lap_times.exists():
                logger.info(f"No lap times found in database. Fetching from Ergast API...")
                lap_number = 1
                while True:
                    lap_url = f"http://ergast.com/api/f1/{race.year}/{race.round_number}/laps/{lap_number}.json"
                    response = requests.get(lap_url)
                    data = response.json()
                    
                    races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
                    if not races:
                        break
                    
                    laps = races[0].get("Laps", [])
                    if not laps:
                        break
                    
                    for lap in laps:
                        for timing in lap.get("Timings", []):
                            if timing["driverId"] == driver.driver_id:
                                lap_time_seconds = convert_lap_time_to_seconds(timing["time"])
                                if lap_time_seconds is not None:
                                    LapTime.objects.create(
                                        race=race,
                                        driver=driver,
                                        lap_number=int(lap["number"]),
                                        lap_time=timing["time"],
                                        lap_time_seconds=lap_time_seconds
                                    )
                                    logger.info(f"Created lap time for lap {lap['number']}: {timing['time']}")
                    
                    lap_number += 1
                
                # Refresh lap times after fetching
                lap_times = LapTime.objects.filter(race=race, driver=driver).order_by('lap_number')
                logger.info(f"Fetched {lap_times.count()} lap times from API")
            
            # Process lap times
            lap_data = []
            total_laps = len(lap_times)
            logger.info(f"Processing {total_laps} laps")
            
            # Get all lap times for prediction
            all_lap_times = [lt.lap_time for lt in lap_times]
            
            for i, lap_time in enumerate(lap_times):
                lap_number = lap_time.lap_number
                actual_time = lap_time.lap_time
                actual_time_seconds = convert_lap_time_to_seconds(actual_time)
                
                logger.info(f"Processing lap {lap_number}: {actual_time} ({actual_time_seconds} seconds)")
                
                if actual_time_seconds is None:
                    logger.warning(f"Skipping lap {lap_number} due to invalid time")
                    continue
                
                # For early laps (1-15), use weighted average
                if lap_number <= 15:
                    # Get all valid lap times up to current lap
                    valid_times = [convert_lap_time_to_seconds(lt) for lt in all_lap_times[:i+1] if convert_lap_time_to_seconds(lt) is not None]
                    
                    if len(valid_times) >= 3:
                        # Use weighted average (more weight to recent laps)
                        weights = np.linspace(0.5, 1.0, len(valid_times))
                        weights = weights / np.sum(weights)
                        predicted_time = np.average(valid_times, weights=weights)
                        logger.info(f"Using weighted average for lap {lap_number}: {predicted_time}")
                    else:
                        # If not enough valid times, use simple average
                        predicted_time = np.mean(valid_times) if valid_times else None
                        logger.info(f"Using simple average for lap {lap_number}: {predicted_time}")
                else:
                    # For later laps, use LSTM
                    predicted_time, _ = predict_next_lap_lstm(
                        all_lap_times[:i+1],
                        lap_number
                    )
                
                # Calculate prediction error if we have the next lap time
                prediction_error = None
                if i < total_laps - 1:
                    next_lap_time = convert_lap_time_to_seconds(lap_times[i+1].lap_time)
                    if predicted_time is not None and next_lap_time is not None:
                        prediction_error = abs(predicted_time - next_lap_time)
                        logger.info(f"Prediction error for lap {lap_number}: {prediction_error}")
                
                # Always add lap data, even if prediction is None
                lap_data.append({
                    'lap_number': lap_number,
                    'lap_time': actual_time,
                    'predicted_next_lap': predicted_time,
                    'prediction_error': prediction_error
                })
            
            # Calculate statistics
            valid_lap_times = [lt for lt in lap_times if convert_lap_time_to_seconds(lt.lap_time) is not None]
            if valid_lap_times:
                fastest_lap = min(valid_lap_times, key=lambda x: convert_lap_time_to_seconds(x.lap_time))
                avg_lap_time = np.mean([convert_lap_time_to_seconds(lt.lap_time) for lt in valid_lap_times])
            else:
                fastest_lap = None
                avg_lap_time = 0.0
            
            prediction_errors = [lap.get('prediction_error', 0) for lap in lap_data if 'prediction_error' in lap and lap['prediction_error'] is not None]
            avg_prediction_error = np.mean(prediction_errors) if prediction_errors else None
            
            # Calculate MSE and RMSE
            mse = np.mean([error ** 2 for error in prediction_errors]) if prediction_errors else None
            rmse = np.sqrt(mse) if mse is not None else None
            
            statistics = {
                'total_laps': total_laps,
                'avg_lap_time': float(avg_lap_time),
                'fastest_lap': {
                    'lap_number': fastest_lap.lap_number if fastest_lap else 0,
                    'time': fastest_lap.lap_time if fastest_lap else "N/A"
                },
                'avg_prediction_error': float(avg_prediction_error) if avg_prediction_error is not None else None,
                'mse': float(mse) if mse is not None else None,
                'rmse': float(rmse) if rmse is not None else None
            }
            
            logger.info(f"Simulation complete. Processed {len(lap_data)} laps")
            
            return JsonResponse({
                'success': True,
                'lap_data': lap_data,
                'statistics': statistics
            })
            
        except Exception as e:
            logger.error(f"Error in LSTM simulation: {str(e)}")
            return JsonResponse({'success': False, 'error': str(e)})
    
    return JsonResponse({'success': False, 'error': 'Invalid request method'}) 