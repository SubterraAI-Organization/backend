from django.contrib import admin

from processing.models import Dataset, Picture, Mask

admin.site.register(Dataset)
admin.site.register(Picture)
admin.site.register(Mask)
