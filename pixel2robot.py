import numpy as np
import json
import function_pool as fp


def pixel2robot(cordx, cordy):
    np.set_printoptions(precision=5)
    np.set_printoptions(suppress=True)
    input_name = "output_wp2camera.json"
    input_name2 = "output_c2f.json"
    input_name3 = "robot_poses.json"
    output_name = "output_b2p.json" #not used
    x = cordx
    y = cordy
    nr = 15
        
    pixel_coords = np.array([[x], [y], [1]])
    tvec, rvec, camera_matrix, dist = fp.read_wp2c(input_name)
    fTc = fp.read_c2f(input_name2)
    bTf_i, _ = fp.read_bTf(input_name3)
    #print("fTc: \n", fTc)
    #print("bTf_i: \n", bTf_i[nr])

    result, bTc, rot_c2p, trans_c2p, bTp, Spitze_mat = fp.calc_pixel2robot(tvec, rvec, camera_matrix, bTf_i, fTc, pixel_coords, nr, output_name, False)#printout results False
    #print(result.shape)
    print("results:")
    result[:3] = result[:3] / 1000
    print(result)

    #XML
    x_robot = result[0, 0]
    y_robot = result[1, 0]
    return x_robot, y_robot, result

# Define the file path
file_path = 'txt_file/center_point.txt'

# Open and read the file
with open(file_path, 'r') as file:
    # Read the content of the file
    content = file.read()

    # Remove parentheses and split the content by comma
    values = content.strip('()').split(',')

    # Convert the values to integers and store them in x and y
    cordx = int(values[0].strip())
    cordy = int(values[1].strip())

x, y, result = pixel2robot(cordx, cordy)

# Your results variable
results = np.array(result)

# Define the file path
file_path = 'txt_file/robot_coordinates.txt'

# Open the file in write mode
with open(file_path, 'w') as file:
    # Iterate over each element in results and write it to the file
    for result in results[:2]:  # Select only the first two elements
        file.write(f"{result[0]:.5f}\n")