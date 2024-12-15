from django.db import models


class Simulation(models.Model):
    option_type = models.CharField(max_length=10, choices=[('call', 'Call'), ('put', 'Put')])
    strike_price = models.FloatField()
    maturity = models.FloatField()  # in years
    volatility = models.FloatField()
    interest_rate = models.FloatField()
    initial_price = models.FloatField()
    num_simulations = models.IntegerField()
    result = models.FloatField(null=True, blank=True)  # Monte Carlo result
    created_at = models.DateTimeField(auto_now_add=True)
