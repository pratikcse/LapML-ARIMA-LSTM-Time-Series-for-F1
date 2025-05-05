from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('simulation/', views.race_simulation, name='race_simulation'),
    path('lstm-simulation/', views.lstm_simulation, name='lstm_simulation'),
    path('fetch-race-data/', views.fetch_race_data, name='fetch_race_data'),
] 