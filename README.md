# F1 Race Simulation

A Django web application that simulates Formula 1 races using historical data from the Ergast API and predicts lap times using ARIMA time series analysis.

## Features

- Fetch and display historical F1 race data
- Select specific races and drivers
- View lap-by-lap times
- Predict next lap times using ARIMA
- Display race statistics including average pace and fastest lap

## Setup

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run migrations:
   ```bash
   python manage.py migrate
   ```

5. Start the development server:
   ```bash
   python manage.py runserver
   ```

6. Open your browser and navigate to `http://127.0.0.1:8000/`

## Usage

1. Select a race from the dropdown menu
2. Select a driver from the dropdown menu
3. Click "Start Simulation" to begin
4. View lap times and predictions in real-time
5. Check the statistics panel for race summary

## Technologies Used

- Django
- Pandas
- NumPy
- Statsmodels (ARIMA)
- Bootstrap 5
- Ergast API

## License

MIT 