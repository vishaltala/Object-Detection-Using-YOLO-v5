import os

# Specify the directory containing your .txt files
directory = 'txt_file'

# Loop through each file in the directory
for filename in os.listdir(directory):
    # Check if the file ends with .txt
    if filename.endswith('.txt'):
        file_path = os.path.join(directory, filename)
        # Open the file in write mode to erase its contents
        with open(file_path, 'w') as file:
            file.write('')  # This will erase the file contents
        print(f"Erased contents of {filename}")
