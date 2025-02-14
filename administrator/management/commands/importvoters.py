import pandas as pd
from django.core.management.base import BaseCommand
from voting.models import Voter
from account.models import CustomUser
import os
from django.conf import settings
import random
import string
import json
from administrator.passwordsofVoters import passwords

def random_password():
    chars = string.ascii_lowercase + string.ascii_uppercase + string.digits
    return ''.join(random.choice(chars) for _ in range(4))

class Command(BaseCommand):
    help = 'Import student from csv file'

    def handle(self, *args, **kwargs):
        # Dictionary to store email-password pairs
        global passwords

        # Define the path to the 'data' folder.
        data_dir = os.path.join(settings.BASE_DIR, 'data')

        # Create the full path to the csv file.
        csv_file_path = os.path.join(data_dir, 'data.csv')

        try:
            # Load the CSV file into a DataFrame
            df = pd.read_csv(csv_file_path)
        except FileNotFoundError:
            self.stdout.write(self.style.ERROR('CSV file not found.'))
            return
        
        passwords.clear() # Clear the passwords dictionary

        for _, row in df.iterrows():
            print(f"Processing: {row['email']}")  # Debugging output
            # Create CustomUser first
            password = row['roll']
            print(f"Paasword: {password}")  # Debugging output
            user = CustomUser.objects.create_user(
                email=row['email'],
                password=row['roll'],
                first_name=row['name'],
                last_name=""
            )
            
            # Store password in dictionary
            passwords[row['roll']] = password
            
            # Create Voter with the user reference
            Voter.objects.create(
                admin=user,
                roll=row['roll']
            )
            print("✅ Import completed successfully!")

        # Save passwords to a JSON file
        passwords_file = os.path.join(data_dir, 'passwords.json')
        with open(passwords_file, 'w') as f:
            json.dump(passwords, f)

        self.stdout.write(self.style.SUCCESS(
            'Successfully imported students'))
        
