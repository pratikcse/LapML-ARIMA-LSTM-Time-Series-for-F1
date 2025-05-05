from django.db import models

class Race(models.Model):
    year = models.IntegerField()
    round_number = models.IntegerField()
    race_name = models.CharField(max_length=200)
    date = models.DateField()
    circuit_name = models.CharField(max_length=200)
    
    def __str__(self):
        return f"{self.year} {self.race_name}"

class Driver(models.Model):
    driver_id = models.CharField(max_length=50, unique=True)
    code = models.CharField(max_length=3)
    first_name = models.CharField(max_length=100)
    last_name = models.CharField(max_length=100)
    
    def __str__(self):
        return f"{self.first_name} {self.last_name}"

class LapTime(models.Model):
    race = models.ForeignKey(Race, on_delete=models.CASCADE)
    driver = models.ForeignKey(Driver, on_delete=models.CASCADE)
    lap_number = models.IntegerField()
    lap_time = models.CharField(max_length=20)
    lap_time_seconds = models.FloatField()
    
    class Meta:
        ordering = ['lap_number']
    
    def __str__(self):
        return f"Lap {self.lap_number} - {self.driver} - {self.lap_time}" 