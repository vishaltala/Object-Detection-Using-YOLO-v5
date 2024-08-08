import numpy as np
import math
import copy
import cv2
from scipy.spatial.transform import Rotation as R
from scipy import linalg
import json
from sympy import degree

"""
Sammlung von Funktion welche im Rahmen einer Kamerakalibrierung am Roboter benötigt werden.
"""
def euler2rot(theta: list, degrees: bool) -> np.array:
    """
    Funktion zum Umrechnen von 3 Winkeln in der Euler-Winkeln-Konvention Roll-Pitch-Yaw in eine 3x3-Rotationsmatrix

    Args:
        theta (list): Liste mit den 3 RPY Winkeln in der Reihenfolgen ZY'X''
        degrees (bool): Gibt an, ob die Input-Winkel in Grad oder rad vorliegen. True = Grad, False = rad

    Returns:
        np.array: 3x3-Rotationsmatrix
    """    
    if degrees:
        alpha = theta[0]*np.pi/180
        beta = theta[1]*np.pi/180
        gamma = theta[2]*np.pi/180
    else:
        alpha = theta[0]
        beta = theta[1]
        gamma = theta[2]
    Rz, Ry, Rx = (np.eye(3, 3) for i in range(3))

    Rz[0, 0] = np.cos(alpha)
    Rz[0, 1] = -np.sin(alpha)
    Rz[1, 0] = np.sin(alpha)
    Rz[1, 1] = np.cos(alpha)

    Ry[0, 0] = np.cos(beta)
    Ry[0, 2] = np.sin(beta)
    Ry[2, 0] = -np.sin(beta)
    Ry[2, 2] = np.cos(beta)

    Rx[1, 1] = np.cos(gamma)
    Rx[1, 2] = -np.sin(gamma)
    Rx[2, 1] = np.sin(gamma)
    Rx[2, 2] = np.cos(gamma)

    return np.linalg.multi_dot((Rz, Ry, Rx))


def rot2euler(rot: np.array, degrees: bool) -> list:
    """
    Funktion zum Umrechnen einer 3x3-Rotationsmatrix in die Euler-Winkel-Konvention Roll-Pitch-Yaw

    Args:
        rot (np.array): 3x3-Rotationsmatrix
        degrees (bool): Gibt an, ob die Output-Winkel in Grad oder rad berechnet werden sollen. True = Grad, False = rad

    Returns:
        list: Liste mit den 3 RPY Winkeln in der Reihenfolgen ZY'X''
    """    
    beta = math.asin(-rot[2, 0])
    if ((beta > (np.pi / 2)) or (beta < -(np.pi / 2))):
        gamma = -(math.atan2(rot[2, 1], rot[2, 2]))
        alpha = -(math.atan2(rot[1, 0], rot[0, 0]))
    else:
        gamma = (math.atan2(rot[2, 1], rot[2, 2]))
        alpha = (math.atan2(rot[1, 0], rot[0, 0]))
    if degrees:
        alpha = alpha*180/np.pi
        beta = beta*180/np.pi
        gamma = gamma*180/np.pi

    return [alpha, beta, gamma]


def read_bTf(input_name: str) -> list[np.array]: 
    """
    Funktion zum Einlesen einer json-Datei mit Roboterposen im Format [x, y, z, a, b, c] und mit abc in der RPY-Konvention

    Args:
        input_name (str): Name der json-Datei mit.json-Endung, welche die Roboterposen enthält. Kann bei abweichendem Dateipfad um diesen ergänzt werden, z.B. "Ordner_xy\posen.json". Die einzelnen Posen stehen unter "Posen" und sind mit "p0" bis "pn" nummeriert. Die einzelnen Werte je Pose stehen unter "x", "y", "z", "a", "b", "c".

    Returns:
        list[np.array]: Liste mit 4x4-Transformationsmatrizen für die Transformationen vom Flansch/TCP-Koordinatensystem ins Basis-KS im Format np.array
    """
    # Daten aus JSON-Datei lesen
    with open(input_name, 'r') as data:
        msg_json = json.load(data)
    
    bTf_i = []
    bTf_i_6D = []
    liste = ["x", "y", "z", "a", "b", "c"]
    for i in range(len(msg_json["Posen"])):
        pose = []
        for j in range(6):
            pose.append(msg_json["Posen"]["p" + str(i)][liste[j]])
        bTf = np.eye(4, 4)
        bTf[0:3, 3] = np.array([pose[0], pose[1], pose[2]])
        theta = [pose[3], pose[4], pose[5]]
        bTf[0:3, 0:3] = euler2rot(theta, degrees=False)
        bTf_i.append(bTf)
        bTf_6D = np.zeros((6,1))
        bTf_6D[0:3, 0] = np.array([pose[0], pose[1], pose[2]])
        bTf_6D[3:6, 0] = np.array([pose[3], pose[4], pose[5]])
        bTf_i_6D.append(bTf_6D)
    
    return bTf_i, bTf_i_6D  
       

def read_b2c(input_name: str) -> np.array:
    """
    Funktion zum Einlesen einer json-Datei mit einer 4x4-Transformationsmatrix zur Transformation von einem ortsfesten Koordianensystem (z.B. ortsfeste Kamera) in das Roboterbasis-KS

    Args:
        input_name (str): Name der json-Datei mit.json-Endung, welche die Transformationsmatrix enthält. Kann bei abweichendem Dateipfad um diesen ergänzt werden, z.B. "Ordner_xy\matrix.json". Die gesuchte Matrix steht unter "bTc(i)" und dann "bTc_average".

    Returns:
        np.array: 4x4-Transformationsmatrix für die Transformation vom Kamera-KS ins Basis-KS
    """    
    with open(input_name, 'r') as f :
        input_params = json.load(f)
    bTc = np.array(input_params["bTc(i)"]["bTc_average"])
    # bTc = np.array(input_params["bTc(i)"]["bTc(5)"])

    return bTc


def read_c2f(input_name: str) -> np.array:
    """
    Funktion zum Einlesen einer json-Datei mit einer 4x4-Transformationsmatrix zur Transforamtion vom Kamera-KS in das Flansch-KS (Kamera fest montiert am Roboter-Flansch).

    Args:
        input_name (str): Name der json-Datei mit.json-Endung, welche die Transformationsmatrix enthält. Kann bei abweichendem Dateipfad um diesen ergänzt werden, z.B. "Ordner_xy\matrix.json". Die gesuchte Matrix steht unter "fTc",

    Returns:
        np.array: 4x4-Transformationsmatrix für die Transformation vom Kamera-KS ins Flansch-KS
    """    
    with open(input_name, 'r') as f :
        input_params = json.load(f)
    fTc = np.array(input_params["fTc"])

    return fTc


def read_wp2c(input_name: str) -> tuple[list, list, np.array, np.array]: 
    """
    Funktion zum Einlesen einer json-Datei mit den Ergebnissen einer Kamerakalibirerung. Hier zu zählen die internen Kameraparameter (Kameramatrix und Distortion-Koeffizienten) und die externen Paramter (Transformationen Schachbrett-KS ins Kamera-KS)
    Kameramatrix: 3x3 Matrix
    Distortion-Koeffizienten: 1x5 Vektor
    Translationsvektor: 3x1 Vektor mit der Position des Schachbrettursprungs im Kamera-KS
    Rotationsvektor: 3x1 Rotationsvektor in der Rodrigues-Konvention, welcher die Drehung des Schachbrett-KS ins Kamera-KS beschreibt


    Args:
        input_name (str): Name der json-Datei mit.json-Endung. Kann bei abweichendem Dateipfad um diesen ergänzt werden, z.B. "Ordner_xy\matrix.json".
                        Kameramatrix: steht unter dem Key "camera-matrix"
                        Distortion-Koeffizienten: steht unter dem Key "dist_coefs"
                        Translationsvektoren: stehen unter dem Key "translational_vectors" und dann unter "image0" usw.
                        Rotationsvektoren: stehen unter dem Key "rotational_vectors" und dann unter "image0" usw.
    Returns:
        tuple[list, list, np.array, np.array]: Tuple bestehend aus Translationsvektoren, Rotationsvektoren, Kameramatrix, Distortion-Koeffizienten 
    """     
    # reads the json file with input parameters
    with open(input_name, 'r') as f :
        input_params = json.load(f)
    camera_matrix = np.array(input_params["camera_matrix"])
    dist = np.array(input_params["dist_coefs"])
    tvec_json = input_params["translational_vectors"]
    rvec_json = input_params["rotational_vectors"]

    tvec  = []
    rvec = []
    for i in range(len(tvec_json)):
        tvec.append(np.array(tvec_json["image" + str(i)]))
        rvec.append(np.array(rvec_json["image" + str(i)]))
    
    return tvec, rvec, camera_matrix, dist


def camera_flange_calibration(tvec: list[np.array], rvec: list[np.array], NoP: int, output_name: str, bTf_i: list[np.array]):
    """
    Funktion zum Berechnen einer Kamera-Flange-Kalibrierung. Ergebnis ist eine 4x4-Transformationsmatrix zur Beschreibung der Transforamtion vom Kamera-KS in das Flansch-KS (Kamera fest montiert am Roboter-Flansch). Implementiert anhand den Ausführungen in TODO

    Args:
        tvec (list[np.array]): 3x1 Vektor mit der Position des Schachbrettursprungs im Kamera-KS
        rvec (list[np.array]): 3x1 Rotationsvektor in der Rodrigues-Konvention, welcher die Drehung des Schachbrett-KS ins Kamera-KS beschreibt
        NoP (int): Anzahl der Bilder, welche für die Kalibrierung benutzt werden sollen. Darf nicht größer sein als die Bilder-Anzahl für die vorangegangene Kamera-Kalibrierung.
        output_name (str): Name der Output-json-Datei, in welcher die Ergebnisse der Kamera-Flange-Kalibrierung abgespeichert werden
        bTf_i (list[np.array]): Liste mit 4x4-Transformationsmatrizen für die Transformationen vom Flansch/TCP-Koordinatensystem ins Basis-KS im Format np.array
    """    
    # NoP = 10    # (NoP = Number of Pictures to be used for the Calibration)
    np.set_printoptions(precision=5)
    np.set_printoptions(suppress=True)

#Input flange coord system related to base coord system (bTf = transformation from base to flange)

    fTf_ges = []
    for i in range(1, NoP):
        fiiTfi = np.dot(np.linalg.inv(bTf_i[i]), bTf_i[0])
        print("fiiTfi: \n", fiiTfi)
        fTf_ges.append(fiiTfi)
    
    # bTf1, bTf2, bTf3, bTf4 = (np.eye(4, 4) for i in range(4))
    # #Translational vector
    # bTf1[0:3, 3] = np.array([488.4, 17.5, -805.1])
    # bTf2[0:3, 3] = np.array([574.5, -10.8, -805.1])
    # bTf3[0:3, 3] = np.array([557.7, -91.1, -790.9])
    # bTf4[0:3, 3] = np.array([498.4, -319.9, -790.9])
    # #rotational vector
    # theta1 = [-179.99, 0.00, -180.00]
    # theta2 = [-180.00, -12.17, -180.00]
    # theta3 = [157.31, -12.18, -179.99]
    # theta4 = [179.61, -1.30, 152.37]
    # bTf1[0:3, 0:3] = euler2rot(theta1, degrees=True)
    # bTf2[0:3, 0:3] = euler2rot(theta2, degrees=True)
    # bTf3[0:3, 0:3] = euler2rot(theta3, degrees=True)
    # bTf4[0:3, 0:3] = euler2rot(theta4, degrees=True)

    
    # f2Tf1 = np.dot(np.linalg.inv(bTf2), bTf1)
    # f3Tf1 = np.dot(np.linalg.inv(bTf3), bTf1)
    # f4Tf1 = np.dot(np.linalg.inv(bTf4), bTf1)
    # fTf_ges = []
    # fTf_ges.append(f2Tf1)
    # fTf_ges.append(f3Tf1)
    # fTf_ges.append(f4Tf1)


# Input chessboard related to camera coord system (cTcb = transformation from camera to chessboard)
    cTcb_i = []
    for i in range(NoP):
        cTcb = np.eye(4, 4)
        cTcb[0:3, 3] = (np.transpose(tvec[i]))
        cv2.Rodrigues(rvec[i], cTcb[0:3, 0:3])
        # print("cTcb: \n", cTcb)
        cTcb_i.append(cTcb)
    
    cTc_ges = []
    for i in range(1, NoP):
        # ciiTci = np.dot(np.linalg.inv(cTcb_i[i]), cTcb_i[0])     #Schachbrett an Flange
        ciiTci = np.dot(cTcb_i[i], np.linalg.inv(cTcb_i[0]))   #Kamera an Flange
        print("ciiTci: \n", ciiTci)
        cTc_ges.append(ciiTci)
    
    fTc, Dx, rx, DA_ges2, DB_ges2, rA_ges, rB_ges= ax_bx_solution(fTf_ges, NoP, cTc_ges)

    print("fTc: \n", fTc)
    output_json = {"Dx": Dx.tolist(), "rx": rx.tolist(), "fTc": fTc.tolist()}
    with open(output_name, "w") as f :
        json.dump(output_json, f, separators=(", ", ":"), indent=2)

# Checking solution
    # translational error
    error1_ges = []
    for i in range(NoP-1):
        e1 = np.dot(DA_ges2[i], rx) + np.transpose(rA_ges[i][np.newaxis]) - np.dot(Dx, np.transpose(rB_ges[i][np.newaxis])) - rx
        error1 = np.linalg.norm(e1)
        error1_ges.append(error1)
    error1_mean = sum(error1_ges)/len(error1_ges) 
    print("translational error: " + str(error1_mean) + " mm")

    # rotational error
    error2_ges = []
    for i in range(NoP-1):
        e2 = np.linalg.multi_dot([DA_ges2[i], Dx, np.linalg.inv(np.dot(Dx, DB_ges2[i]))])
        error2 = np.linalg.norm(cv2.Rodrigues(e2)[0])
        error2_ges.append(error2)
    error2_mean = sum(error2_ges)/len(error2_ges)
    print("rotational error: " + str(error2_mean) + " rad")
    print("rotational error: " + str(error2_mean*180/np.pi) + " degrees")
    
    

def ax_bx_solution(fTf_ges: list[np.array], NoP: int, cTc_ges: list[np.array]) -> tuple[np.array, np.array, np.array, list[np.array], list[np.array], list[np.array], list[np.array]]:
    """
    Funktion zum Lösen eines nichtlinearen Gleichungssystems der Form AX = XB

    Args:
        fTf_ges (list[np.array]): _description_
        NoP (int): _description_
        cTc_ges (list[np.array]): _description_

    Returns:
        tuple[np.array, np.array, np.array, list[np.array], list[np.array], list[np.array], list[np.array]]: _description_
    """    
# Solution for equation AX = XB
    DA_ges, DB_ges = (None for i in range(2))
    DA_ges2, rA_ges,DB_ges2, rB_ges = ([] for i in range(4))
    for i in range(0, NoP-1):
        DAi_mat = fTf_ges[i][0:3, 0:3]
        DA_ges2.append(DAi_mat)
        DAi = R.from_matrix(DAi_mat)
        DAi_rvec = DAi.as_rotvec()
        # DA_ges.append(DAi_rvec)

        DBi_mat = cTc_ges[i][0:3, 0:3]
        DB_ges2.append(DBi_mat)
        DBi = R.from_matrix(DBi_mat)
        DBi_rvec = DBi.as_rotvec()
        # DB_ges.append(DBi_rvec)
        
        if i == 0:
            DA_ges = DAi_rvec
            DB_ges = DBi_rvec
        else:
            DA_ges = np.c_[DA_ges, DAi_rvec]
            DB_ges = np.c_[DB_ges, DBi_rvec]
        

        rAi = fTf_ges[i][0:3, 3]
        rA_ges.append(rAi)

        rBi = cTc_ges[i][0:3, 3]
        rB_ges.append(rBi)


    T = np.dot(DB_ges, np.transpose(DA_ges))

# singular value decomposition
    U, S, V = linalg.svd(T)  # V is the tranpose of the matrix calculated by matlab svd
    Up = np.dot(U, V)
    P = np.linalg.multi_dot((np.transpose(V), np.diag(S), V))
    USp, Dp, Vp = linalg.svd(P)  # Vp is the tranpose of the matrix calculated by matlab svd
    #-----------------------------------------------
    row, col = np.shape(P)
    f = np.linalg.det(Up)
    if f<0:
        X = np.eye(row, col) * (-1)
    else:
        X = np.eye(row, col)
# calculate rotation matrix Dx 
    Dx = np.linalg.multi_dot([np.transpose(Vp), X, Vp, np.transpose(Up)])
    print ("Dx: \n", Dx)
# calculate translational vector rx
    E = np.eye(3, 3)
    C_ges, d_ges = (None for i in range(2))
    for i in range(NoP-1):
        C = np.subtract(E, fTf_ges[i][0:3, 0:3])
        d = np.subtract(rA_ges[i], np.dot(Dx, rB_ges[i]))
        # C_ges.append(C)

        if i == 0:
            C_ges = C
            d_ges = np.array(np.transpose([d])) 
        else:
            C_ges = np.vstack((C_ges, C))
            d_ges = np.vstack((d_ges, np.array(np.transpose([d]))))
    
    rx = np.dot(np.linalg.pinv(C_ges), d_ges)
    print("rx: \n", rx)

    fTc = np.eye(4, 4)
    fTc[0:3, 0:3] = Dx
    fTc[0:3, 3] = np.transpose(rx)

    return fTc, Dx, rx, DA_ges2, DB_ges2, rA_ges, rB_ges



def camera_base_calibration(tvec: np.array, rvec, NoP, bTf_i, fTc):
    """
    Args:
        tvec (np.array): _description_
        rvec (_type_): _description_
        NoP (_type_): _description_
        bTf_i (_type_): _description_
        fTc (_type_): _description_

    Returns:
        _type_: _description_
    """    
# transformation robot base to camera, if camera is not attached on robot flange
    cTcb_i = []
    for i in range(NoP):
        cTcb = np.eye(4, 4)
        cTcb[0:3, 3] = (np.transpose(tvec[i]))
        cv2.Rodrigues(rvec[i], cTcb[0:3, 0:3])
        # print("cTcb: \n", cTcb)
        cTcb_i.append(cTcb)
    
    cTc_ges = []
    for i in range(1, NoP):
        # ciiTci = np.dot(np.linalg.inv(cTcb_i[i]), cTcb_i[0])     #Schachbrett an Flange
        ciiTci = np.dot(cTcb_i[i], np.linalg.inv(cTcb_i[0]))   #Kamera an Flange
        print("ciiTci: \n", ciiTci)
        cTc_ges.append(ciiTci)
    bTc_i = []

    output_json2 = {"bTc(i)": {}}
    for i in range(len(bTf_i)):
        # bTc = np.linalg.multi_dot([bTf_i[i], fTc, cTcb_i[i]])               #Kamera an Flange
        bTc = np.linalg.multi_dot([bTf_i[i], fTc, np.linalg.inv(cTcb_i[i])])  #Schachbrett an Flange
        output_json2["bTc(i)"].update({"bTc("+str(i+1)+")": bTc.tolist()})
        bTc_i.append(bTc)
    average = sum(bTc_i)/len(bTc_i)
    output_json2["bTc(i)"].update({"bTc_average": average.tolist()})
    print("bTc(1): \n", bTc_i[0])
    print("average: \n", average)
    with open("output_b2c.json", "w") as f :
            json.dump(output_json2, f, separators=(", ", ":"), indent=2)

    # comparison average
    bTcb_i = None
    cTcb_i_2 = None
    for i in range(len(bTf_i)):
        bTcb = np.dot(bTf_i[i], fTc)

        if i == 0:
            bTcb_i = bTcb
            cTcb_i_2 = cTcb_i[i]
        else:
            bTcb_i = np.c_[bTcb_i, bTcb]  
            cTcb_i_2 = np.r_[cTcb_i_2, cTcb_i[i]] 
    
    average2 = np.dot(bTcb_i, np.linalg.pinv(cTcb_i_2))
    print("average2: \n", average2)

def calc_pixel2robot(tvec, rvec, camera_mat, bTf, fTc, pixel_coords, nr, output_name, printout):
    np.set_printoptions(precision=5)
    np.set_printoptions(suppress=True)

    # pixel_tmp = np.array([[pixel_coords[nr][-1][0][0]], [pixel_coords[nr][-1][0][1]], [1]])
    pixel_tmp = pixel_coords
    print("pixel_coords:\n", pixel_tmp)
    rot_mat = np.zeros((3, 3))
    cv2.Rodrigues(rvec[nr], rot_mat)
    # print("rot_mat:\n", rot_mat)
    # cam_flange_trans = np.array([[0.0119, -0.9993, 0.0351, 64.6819], [-0.9999, -0.0121, -0.0075, -6.9471], [0.0080, -0.0350, -0.9994, 67.6123], [0, 0, 0, 1]])
    cam_flange_trans = fTc

    flange_base_trans = bTf[nr]
    # flange_base_trans = np.zeros((4,4))
# Picture p01
    # flange_base_trans[:, 3] = np.array([488.39, 17.55, -805.12, 1])
    # rot = R.from_euler('zyx', ([-179.99, 0.00, -180.00]), degrees=True)
    # flange_base_trans[0:3, 0:3] = rot.as_matrix()
# Picture p04
    # flange_base_trans[:, 3] = np.array([498.43, -319.86, -790.91, 1])
    # rot = R.from_euler('zyx', ([-179.61, -1.30, -152.37]), degrees=True)
    # flange_base_trans[0:3, 0:3] = rot.as_matrix()

    a_b = np.dot(np.linalg.inv(camera_mat), pixel_tmp)
    # print("a_b: \n", a_b)
    matrix = np.zeros((3, 3))
    matrix[0:3, 0:2] = -rot_mat[0:3, 0:2]
    matrix[0:3, 2] = np.transpose(a_b)

    xp_yp_zc = np.dot(np.linalg.inv(matrix), tvec[nr])
    # print("xp_yp_zc1: \n", xp_yp_zc)
    xp_yp_zc[0] = a_b[0]*xp_yp_zc[2]
    xp_yp_zc[1] = a_b[1]*xp_yp_zc[2]
    # xp_yp_zc[0] = rot_mat[0][0]*xp_yp_zc[0]+rot_mat[0][1]*xp_yp_zc[1]+tvec[nr][0]
    # xp_yp_zc[1] = rot_mat[1][0]*xp_yp_zc[0]+rot_mat[1][1]*xp_yp_zc[1]+tvec[nr][1]
    # print("xp_yp_zc2: \n", xp_yp_zc)
    
    # transformation of any pixel in to robot base coordinate system
    trans_result = np.dot(flange_base_trans, cam_flange_trans)
    pixel_coord = np.zeros((4, 1))
    pixel_mat = np.eye(4,4)
    pixel_coord[0:3, 0] = np.transpose(xp_yp_zc)
    pixel_coord[3] = [1]
    pixel_mat[0:3, 3] = np.transpose(xp_yp_zc)
    pixel_mat[0:3, 0:3] = rot_mat
    result = np.dot(trans_result, pixel_coord)
    result2 = np.dot(trans_result, pixel_mat)
    

    # robot flange positon to reach the selected point (pixel) with the TCP of an attached tool
    Spitze_mat = np.eye(4, 4)
    Spitze_mat[2,3] = 112
    result3 = np.dot(result2, np.linalg.inv(Spitze_mat))
    result4 = np.zeros((6,1))
    result4[0:3, 0] = result3[0:3, 3]
    rpy = rot2euler(result3[0:3, 0:3], degrees=True)
    result4[3:7, 0] = np.transpose(rpy)
    
    
    if printout:
        print("Transformation-Matrix: \n", trans_result)
        print("result: \n", result)
        print("result2: \n", result2)
        print("pixel_mat: \n", pixel_mat)
        print("Spitze: \n", Spitze_mat)
        print("result3: \n", result3)
        print("result4: \n", result4)

    # output_json = {"bTc"+str(nr): trans_result.tolist(), "fTs": Spitze_mat.tolist(), "rot_c2p": rot_mat.tolist(), "trans_c2p": xp_yp_zc.tolist(), "bTp": result2.tolist()}
    # with open(output_name, "w") as f :
    #     json.dump(output_json, f, separators=(", ", ":"), indent=2)

    return result, trans_result, rot_mat, xp_yp_zc, result2, Spitze_mat

def pixel2robot_tangram(tvec, rvec, camera_mat, bTc, pixel_coords, nr):
    np.set_printoptions(precision=5)
    np.set_printoptions(suppress=True)

    pixel_tmp = pixel_coords

    rot_mat = np.zeros((3, 3))
    cv2.Rodrigues(rvec[nr], rot_mat)

    a_b = np.dot(np.linalg.inv(camera_mat), pixel_tmp)

    matrix = np.zeros((3, 3))
    matrix[0:3, 0:2] = -rot_mat[0:3, 0:2]
    matrix[0:3, 2] = np.transpose(a_b)

    xp_yp_zc = np.dot(np.linalg.inv(matrix), tvec[nr])
    xp_yp_zc[0] = a_b[0]*xp_yp_zc[2]
    xp_yp_zc[1] = a_b[1]*xp_yp_zc[2]

    pixel_mat = np.eye(4,4)
    pixel_mat[0:3, 3] = np.transpose(xp_yp_zc)
    pixel_mat[0:3, 0:3] = rot_mat

    result = np.dot(bTc, pixel_mat)
    result2 = np.zeros((6,1))
    result2[0:3, 0] = result[0:3, 3]
    rpy = rot2euler(result[0:3, 0:3], degrees=True)
    result2[3:7, 0] = np.transpose(rpy)

    # robot flange positon to reach the selected point (pixel) with the TCP of an attached tool
    Spitze_mat = np.eye(4, 4)
    Spitze_mat[2,3] = 118           #116.5
    result3 = np.dot(result, np.linalg.inv(Spitze_mat))
    result4 = np.zeros((6,1))
    result4[0:3, 0] = result3[0:3, 3]
    rpy = rot2euler(result3[0:3, 0:3], degrees=True)
    result4[3:7, 0] = np.transpose(rpy)

    print("result: \n", result)
    print("result2: \n", result2)
    print("result4: \n", result4)
    
def undistort_pixel(dist_coeff, camera_matrix, pixel_coords, nr):
    print("pixel_coord:\n", pixel_coords)
    # after first camera calibration cam_mtx and distCoff, pixel_coords in form np.array([[[x, y]]], np.float32)
    undist_pixel = cv2.undistortPoints(pixel_coords, camera_matrix, dist_coeff)
    print("new pixel_coord:\n", undist_pixel[0][0])
    pix1 = cv2.convertPointsToHomogeneous(undist_pixel)[0][0] 
    print("pix1:\n", pix1)
    pix = np.dot(camera_matrix, np.transpose(pix1[np.newaxis]))
    print("pix:\n", pix)
    return

