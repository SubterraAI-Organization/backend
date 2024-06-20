# processing/management/commands/createsuperuser_if_not_exists.py

from django.core.management.base import BaseCommand
from django.contrib.auth.models import User

class Command(BaseCommand):
    help = 'Create a superuser if it does not exist'

    def handle(self, *args, **options):
        username = 'shakoor'
        email = 'shakoor@example.com'
        password = 'ShakoorLab@2023'

        if not User.objects.filter(username=username).exists():
            User.objects.create_superuser(username=username, email=email, password=password)
            self.stdout.write(self.style.SUCCESS(f'Successfully created superuser: {username}'))
        else:
            self.stdout.write(self.style.SUCCESS(f'Superuser {username} already exists'))

