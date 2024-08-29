import socket
import struct
import time
import subprocess
import os

def send_urscript(script, host):
    port = 30002  # UR secondary client port
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))
    s.send(script.encode('utf-8'))
    s.close()

def generate_urscript_movej(x, y, z, rx, ry, rz, a=0.1, v=0.1):
    urscript = f"""
def move_to_position():
    movej(p[{x}, {y}, {z}, {rx}, {ry}, {rz}], a={a}, v={v})
    textmsg("Movement complete!")
end

move_to_position()
"""
    return urscript

def generate_urscript_movel(x, y, z, rx, ry, rz, a=0.1, v=0.1):
    urscript = f"""
def move_to_position():
    movel(p[{x}, {y}, {z}, {rx}, {ry}, {rz}], a={a}, v={v})
    textmsg("Movement complete!")
end

move_to_position()
"""
    return urscript

def generate_urscript_movej_forward(x, y, z, rx, ry, rz, a=1, v=1):
    urscript = f"""
def move_to_position():
    movej([{x}, {y}, {z}, {rx}, {ry}, {rz}], a={a}, v={v})
    textmsg("Movement complete!")
end

move_to_position()
"""
    return urscript

def generate_urscript_movel_forward(x, y, z, rx, ry, rz, a=1, v=1):
    urscript = f"""
def move_to_position():
    movel([{x}, {y}, {z}, {rx}, {ry}, {rz}], a={a}, v={v})
    textmsg("Movement complete!")
end

move_to_position()
"""
    return urscript

def suction_on(robot_ip):
    urscript = f"""
def start_suction():
    set_tool_digital_out(0, False)  # Turn off the vacuum pump (using digital output 1 of the tool output)
    sleep(2)  # Wait for 2 seconds
    set_tool_digital_out(1, True)  # Turn on the vacuum pump (using digital output 1 of the tool output)
    textmsg("Vacuum pump started!")
end

start_suction()
"""
    send_urscript(urscript, robot_ip)

def suction_off(robot_ip):
    urscript = f"""
def start_release():
    set_tool_digital_out(1, False)  # Turn off the vacuum pump (using digital output 1 of the tool output)
    sleep(2)  # Wait for 2 seconds
    set_tool_digital_out(0, True)  # Turn on the vacuum pump (using digital output 1 of the tool output)
    textmsg("Vacuum pump started!")
end

start_release()
"""
    send_urscript(urscript, robot_ip)

def get_current_position(robot_ip):
    port = 30003  # Real-time data port
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((robot_ip, port))
    data = s.recv(1108)
    s.close()
    
    # Extract the position data from the received packet correctly
    unpacked_data = struct.unpack('!6d', data[444:492])  # 6 double values starting at byte 444
    x, y, z, rx, ry, rz = unpacked_data
    
    return [x, y, z, rx, ry, rz]

def get_joint_angles(robot_ip):
    PORT = 30003  # UR robots typically use port 30003 for primary interface
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((robot_ip, PORT))
    data = s.recv(2048)
    s.close()
    
    # Extract the position data from the received packet correctly
    joint_angles = struct.unpack('!6d', data[252:300])
    base, shoulder, elbow, wrist1, wrist2, wrist3 = joint_angles

    return [base, shoulder, elbow, wrist1, wrist2, wrist3]

def has_reached_position(current_pos, target_pos, threshold=0.005):
    # Compare only the x, y, z coordinates
    for current, target in zip(current_pos[:3], target_pos[:3]):
        if abs(current - target) > threshold:
            return False
    return True

def move_to_main_position(robot_ip):
    main_position = [-0.2553, -1.6563, 0.9641, -0.8604, -1.5900, 2.0661]
    script = generate_urscript_movej_forward(*main_position)
    send_urscript(script, robot_ip)

     # Wait until the robot reaches the position
    while True:
        current_position = get_joint_angles(robot_ip)
        if has_reached_position(current_position, main_position):
            break
        time.sleep(0.1)

x_robot = None
y_robot = None

def move_to_object(robot_ip):
    global x_robot, y_robot
    with open('txt_file/robot_coordinates.txt', 'r') as file:
        coordinates = file.readlines()
        x_robot = float(coordinates[0].strip())
        x_robot_offset = x_robot + 0.084  # Offset for Camera and Tool
        y_robot = float(coordinates[1].strip())

    new_position = [x_robot_offset, y_robot, 0.5, 2.944, -1.163, 0.023]
    script = generate_urscript_movel(*new_position)
    send_urscript(script, robot_ip)

    # Wait until the robot reaches the new position
    while True:
        current_position = get_current_position(robot_ip)
        if has_reached_position(current_position, new_position):
            break
        time.sleep(0.1)

def pick_the_object(robot_ip):
    file_path = 'txt_file/robot_RPY.txt'

    # Open and read the file
    with open(file_path, 'r') as file:
        content = file.read().strip()

    # Parse the content
    values = list(map(float, content.strip('[]').split()))
    rx, ry = values[0], values[1]

    pick_position = [x_robot, y_robot, z_pick, rx, ry, 0.0]
    script = generate_urscript_movel(*pick_position)
    send_urscript(script, robot_ip)

    # Wait until the robot reaches the pick position
    while True:
        current_position = get_current_position(robot_ip)
        if has_reached_position(current_position, pick_position):
            break
        time.sleep(0.1)
    time.sleep(1)

def pick_up_object(robot_ip):
    file_path = 'txt_file/robot_RPY.txt'

    # Open and read the file
    with open(file_path, 'r') as file:
        content = file.read().strip()

    # Parse the content
    values = list(map(float, content.strip('[]').split()))
    rx, ry = values[0], values[1]

    pick_up_position = [x_robot, y_robot, 0.2, rx, ry, 0.0]
    script = generate_urscript_movel(*pick_up_position)
    send_urscript(script, robot_ip)

    # Wait until the robot reaches the pick position
    while True:
        current_position = get_current_position(robot_ip)
        if has_reached_position(current_position, pick_up_position):
            break
        time.sleep(0.1)
    time.sleep(1)

def intermediate_position(robot_ip):
    intermediate_position = [-1.8497, -1.8064, 1.9345, -1.6989, -1.5687, -1.8500]
    script = generate_urscript_movej_forward(*intermediate_position)
    send_urscript(script, robot_ip)

     # Wait until the robot reaches the position
    while True:
        current_position = get_joint_angles(robot_ip)
        if has_reached_position(current_position, intermediate_position):
            break
        time.sleep(0.1)

def final_position(robot_ip):
    final_position = [x_place, y_place, z_place, 2.221, 2.221, 0]
    script = generate_urscript_movel(*final_position)
    send_urscript(script, robot_ip)

    # Wait until the robot reaches the pick position
    while True:
        current_position = get_current_position(robot_ip)
        if has_reached_position(current_position, final_position):
            break
        time.sleep(0.1)

def detect_object():
    subprocess.run(["python", "detection.py"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def convert_pixel_to_robot():
    subprocess.run(["python", "pixel2robot.py"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def pca_calculation():
    subprocess.run(["python", "pca.py"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def direction_object():
    subprocess.run(["python", "direction.py"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def delet_txt_file():
    directory = 'txt_file'
    # Loop through each file in the directory
    for filename in os.listdir(directory):
        # Check if the file ends with .txt
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            # Open the file in write mode to erase its contents
            with open(file_path, 'w') as file:
                file.write('')  # This will erase the file contents


# Robot IP address
robot_ip = '192.168.0.118'

# Execute the workflow

while True:
    move_to_main_position(robot_ip)
    detect_object()

    file_path = 'txt_file/label.txt'
    with open(file_path, 'r') as file:
        contents = file.read().strip()
        if len(contents) == 0:
            print("Object has not been found")
            break
    
    convert_pixel_to_robot()
    move_to_object(robot_ip)
    detect_object()

    # Define the file path
    file_path = 'txt_file/label.txt'
    # Open and read the file
    with open(file_path, 'r') as file:
        content = file.read().strip()
    # Split the content and get the first element
    object_type = int(content.split()[0])

    # Declare the variables globally at the start
    z_pick = None
    x_place, y_place, z_place = None, None, None
    if object_type == 0:
        z_pick = 0.05
        x_place, y_place, z_place = 0.366, -0.146, 0.067
    elif object_type == 1:
        z_pick = 0.030
        x_place, y_place, z_place = 0.366, -0.008, 0.048
    elif object_type == 2:
        z_pick = 0.020
        x_place, y_place, z_place = 0.366, 0.121, 0.040
    else:
        raise ValueError("Invalid object_type. It must be 0, 1, or 2.")

    pca_calculation()
    direction_object()
    pick_the_object(robot_ip)
    suction_on(robot_ip)
    time.sleep(3)
    pick_up_object(robot_ip)
    intermediate_position(robot_ip)
    final_position(robot_ip)
    suction_off(robot_ip)
    time.sleep(3)
    intermediate_position(robot_ip)
    move_to_main_position(robot_ip)

    # Erase all previous data
    delet_txt_file()