from django.contrib import admin
from .models import Movie, UserRating, UserList

# Register your models here.
admin.site.register(Movie)
admin.site.register(UserRating)
admin.site.register(UserList)