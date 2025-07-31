import os

# Execute the command "ls -l" using os.system()
os.system('ls -l')

import os

# Execute the command "ls -l" using os.system()
status = os.system('ls -l')

if status == 0:
    print("Command executed successfully!")
else:
    print("Command failed to execute!")