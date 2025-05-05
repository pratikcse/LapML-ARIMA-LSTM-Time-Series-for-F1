from django.core.management.base import BaseCommand
from race.models import Race, Driver, LapTime
import requests
from datetime import datetime

class Command(BaseCommand):
    help = 'Populates the database with initial race and driver data'

    def handle(self, *args, **kwargs):
        # Fetch current season data
        current_year = 2023
        base_url = f"http://ergast.com/api/f1/{current_year}.json"
        
        try:
            # Fetch race data
            response = requests.get(base_url)
            data = response.json()
            races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
            
            for race_data in races:
                race, created = Race.objects.get_or_create(
                    year=current_year,
                    round_number=race_data.get("round"),
                    defaults={
                        'race_name': race_data.get('raceName', ''),
                        'date': datetime.strptime(race_data.get('date', ''), '%Y-%m-%d').date(),
                        'circuit_name': race_data.get('Circuit', {}).get('circuitName', '')
                    }
                )
                if created:
                    self.stdout.write(self.style.SUCCESS(f'Created race: {race.race_name}'))
            
            # Fetch driver data
            drivers_url = f"http://ergast.com/api/f1/{current_year}/drivers.json"
            drivers_response = requests.get(drivers_url)
            drivers_data = drivers_response.json()
            
            for driver_data in drivers_data.get("MRData", {}).get("DriverTable", {}).get("Drivers", []):
                driver, created = Driver.objects.get_or_create(
                    driver_id=driver_data.get('driverId', ''),
                    defaults={
                        'code': driver_data.get('code', ''),
                        'first_name': driver_data.get('givenName', ''),
                        'last_name': driver_data.get('familyName', '')
                    }
                )
                if created:
                    self.stdout.write(self.style.SUCCESS(f'Created driver: {driver.first_name} {driver.last_name}'))
            
            self.stdout.write(self.style.SUCCESS('Successfully populated the database'))
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Error populating database: {str(e)}')) 