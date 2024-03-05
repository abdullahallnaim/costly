from django.db import models

# Create your models here.

foundations = [
    ['Deep/Pile', 0],
    ['Shallow (without footing)', 1],
    ['Deep/Pile', 2],
]

class PredictCost(models.Model):
    floors = models.IntegerField()
    ordinary_rooms = models.IntegerField()
    kitchens = models.IntegerField()
    toilets = models.IntegerField()
    columns = models.IntegerField()
    building_height = models.FloatField()
    floor_area = models.FloatField()
    brickwall_area = models.FloatField()
    type_of_foundation = models.CharField(choices = foundations, max_length = 100)