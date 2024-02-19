import subprocess
from fabric import Connection

#Transfer directory to Solstorm
def transfer_directory(local_directory):
    """
    Transfer a directory to a remote Unix-based cluster using SCP.

    Args:
    - local_directory (str): The path to the local directory to transfer.

    Consts:
    - remote_directory (str): The path to the remote directory where the local directory will be transferred.
    - username (str): The username for the remote system.
    - host (str): The hostname or IP address of the remote system.
    """
    username = "olavsbj"
    host = "solstorm-login.iot.ntnu.no"
    remote_directory = "/mnt/beegfs/users/olavsbj"

    try:
        # Construct the SCP command
        scp_command = f"scp -r {local_directory} {username}@{host}:{remote_directory}"

        # Execute the SCP command
        subprocess.check_call(scp_command, shell=True)
        print("Directory successfully transferred.")
    except subprocess.CalledProcessError as e:
        print(f"Error during directory transfer: {e}")

#Run file in Solstorm:


def run_file(path : str, screen_name : str, node : list):
    #Creating connection object
    print("ok1")
    c = Connection('olavsbj@solstorm-login.iot.ntnu.no')

    # Create screen
    print("ok2")
    _ = c.run(f"screen -dmS {screen_name}")

    # Log in to specified node
    print("ok3")
    _ = c.run(f"ssh compute-{node[0]}-{node[1]}", pty=True)

# Navigate to root folder
    print("ok4")
    _ = c.run(f"cd ~")

    # Loading modules
    print("ok5")
    _ = c.run(f"module load gurobi/11.0")
    _ = c.run(f"module load Python/3.11.5-GCCcore-13.2.0")

    # Running file
    file_path = f"storage/users/olavsbj/" + path

    try:
        result = c.run(f"python3 {file_path}", pty=True)
    except:
        print(result)



if __name__ == "__main__":
    #Setting the local directory that is to be use
    local_directory_path = "/"
    #transfer_directory(local_directory_path)

    # Run the file
    run_file("tpppsf-L-shaped/test-script.py", "test-screen-1", [4,21])



