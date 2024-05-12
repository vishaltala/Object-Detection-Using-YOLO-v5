import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Given points
points = np.array(
    [[211, 333], [450, 78], [796, 129], [786, 369], [744, 122], [689, 117], [747, 411], [433, 72], [778, 127], [326, 351], [781, 375], [85, 280], [594, 97], [773, 127], [663, 399], [614, 102], [452, 78], [762, 126], [483, 81], [654, 399], [765, 126], [356, 354], [574, 94], [692, 404], [62, 160], [824, 167], [649, 399], [77, 241], [71, 214], [694, 116], [782, 372], [200, 34], [485, 374], [622, 103], [437, 75], [213, 333], [514, 379], [60, 175], [587, 97], [672, 115], [523, 380], [176, 31], [147, 27], [559, 385], [334, 351], [813, 269], [131, 323], [655, 399], [824, 163], [359, 61], [497, 378], [82, 249], [59, 248], [822, 154], [307, 345], [127, 24], [284, 343], [108, 41], [364, 357], [240, 337], [372, 358], [454, 78], [216, 12], [93, 69], [102, 312], [768, 402], [67, 95], [106, 41], [297, 51], [665, 400], [215, 39], [82, 264], [732, 419], [226, 39], [818, 253], [88, 79], [192, 332], [58, 106], [408, 363], [399, 362], [649, 106], [406, 362], [248, 339], [771, 395], [757, 413], [726, 443], [582, 389], [503, 378], [575, 386], [323, 55], [824, 170], [482, 81], [52, 142], [77, 240], [47, 201], [60, 217], [444, 78], [53, 231], [808, 131], [468, 372], [62, 158], [111, 30], [474, 374], [726, 442], [98, 309], [554, 385], [78, 254], [45, 178], [381, 65], [60, 160], [53, 230], [265, 340], [194, 34], [705, 116], [340, 351], [320, 55], [677, 402], [284, 49], [194, 36], [449, 370], [630, 396], [582, 95], [64, 184], [807, 297], [654, 108], [368, 63], [251, 43], [369, 358], [731, 119], [96, 59], [62, 167], [60, 203], [88, 305], [799, 322], [541, 89], [60, 187], [588, 97], [687, 404], [717, 408], [753, 124], [813, 267], [797, 129], [789, 127], [60, 212], [33, 175], [532, 89], [748, 122], [43, 177], [105, 43], [733, 416], [424, 72], [560, 94], [77, 245], [772, 394], [82, 272], [768, 403], [110, 316], [291, 343], [140, 323], [492, 83], [542, 89], [221, 336], [308, 54], [726, 441], [93, 65], [791, 127], [818, 226], [611, 101], [577, 387], [362, 62], [54, 116], [637, 105], [160, 30], [798, 330], [360, 356], [60, 216], [525, 87], [764, 409], [725, 437], [172, 328], [206, 332], [197, 36], [808, 294], [606, 101], [45, 186], [666, 108], [70, 92], [77, 248], [83, 86], [146, 323], [104, 313], [60, 222], [82, 254], [440, 75], [170, 328], [45, 200], [27, 164], [68, 199], [62, 166], [572, 94], [147, 323], [60, 196], [60, 210], [561, 384], [789, 357], [794, 346], [52, 128], [60, 194], [180, 330], [367, 63], [82, 262], [610, 392], [82, 269], [218, 12], [472, 374], [815, 264], [40, 153], [61, 241], [724, 119], [49, 213], [83, 275], [412, 69], [556, 384], [341, 351], [47, 208], [93, 66], [821, 191], [64, 188], [84, 81], [62, 164], [57, 114], [664, 108], [224, 336], [121, 320], [622, 394], [53, 122], [209, 39], [60, 201], [644, 106], [197, 34], [256, 340], [53, 238], [732, 428], [35, 177], [130, 323], [250, 43], [246, 339], [273, 47], [119, 24], [77, 252], [660, 108], [193, 332], [822, 155], [93, 67], [506, 85], [753, 412], [769, 126], [734, 119], [140, 25], [546, 91], [69, 212], [208, 39], [822, 160], [84, 283], [494, 83], [764, 126], [823, 174], [165, 327], [431, 367], [93, 68], [792, 350], [634, 396], [725, 119], [505, 85], [418, 71], [62, 244], [49, 201], [44, 177], [535, 382], [73, 215], [89, 305], [103, 48], [253, 43], [75, 221], [264, 45], [497, 376], [111, 32], [75, 220], [398, 67], [45, 204], [494, 375], [45, 184], [823, 177], [56, 115], [806, 129], [289, 343], [277, 48], [60, 218], [279, 49], [300, 343], [540, 383], [62, 180], [327, 57], [479, 374], [449, 78], [809, 288], [305, 53], [331, 60], [682, 120], [808, 292], [641, 396], [498, 84], [62, 150], [255, 340], [139, 25], [117, 319], [71, 213], [129, 320], [594, 390], [155, 326], [444, 368], [480, 374], [60, 176], [266, 340], [573, 386], [824, 179], [73, 92], [434, 367], [570, 386], [49, 207], [495, 83], [732, 411], [810, 280], [84, 277], [205, 31], [399, 67], [53, 240], [762, 413], [596, 390], [60, 249], [228, 336], [798, 329], [49, 222], [143, 323], [614, 393], [384, 65], [707, 405], [52, 130], [53, 235], [500, 378], [400, 362], [821, 189], [515, 86], [122, 23], [633, 396], [783, 371], [715, 117], [357, 354], [45, 185], [113, 317], [585, 97], [509, 378], [564, 385], [446, 369], [125, 23], [821, 186], [816, 262], [229, 336], [627, 396], [732, 119], [123, 23], [410, 363], [49, 144], [60, 229], [818, 227], [437, 367], [328, 56], [522, 380], [202, 332], [53, 118], [434, 73], [101, 311], [332, 60], [818, 216], [294, 51], [38, 177], [178, 31], [49, 220], [118, 320], [82, 260], [61, 98], [391, 66], [390, 65], [323, 350], [247, 43], [60, 138], [818, 231], [49, 209], [448, 78], [645, 396], [730, 409], [157, 30], [821, 208], [82, 252], [708, 405], [646, 398], [111, 316], [259, 340], [741, 411], [158, 30], [413, 69], [771, 126], [210, 333], [819, 214], [823, 161], [45, 179], [84, 292], [60, 214], [818, 233], [151, 29], [39, 153], [60, 173], [49, 210], [49, 224], [62, 165], [291, 51], [480, 80], [776, 126], [609, 101], [368, 358], [98, 54], [49, 214], [433, 367], [643, 106], [287, 49], [802, 129], [499, 84], [82, 251], [234, 41], [29, 170], [83, 295], [309, 54], [185, 330], [613, 393], [797, 334], [56, 243], [695, 404], [68, 206], [733, 119], [558, 94], [639, 396], [671, 402], [782, 127], [104, 45], [423, 366], [61, 238], [751, 123], [138, 323], [511, 86], [183, 31], [517, 379], [665, 108], [811, 132], [53, 119], [87, 79], [217, 335], [711, 408], [493, 374], [590, 390], [307, 54], [673, 112], [733, 421], [91, 72], [322, 55], [324, 350], [394, 67], [202, 39], [760, 413], [604, 100], [475, 374], [62, 177], [733, 425], [418, 365], [68, 211], [60, 145], [768, 126], [45, 197], [393, 362], [55, 241], [813, 132], [752, 412], [603, 391], [60, 159], [565, 385], [416, 71], [675, 402], [808, 295], [554, 92], [63, 142], [45, 199]]
)

# Fit PCA
pca = PCA(n_components=2)
pca.fit(points)

# First principal component
first_pc = pca.components_[0]

# Center of all points (mean)
center_point = np.mean(points, axis=0)

# Plotting the points and the principal component
plt.figure(figsize=(8, 6))
plt.scatter(points[:, 0], points[:, 1], alpha=0.7)
for length, vector in zip(pca.explained_variance_, pca.components_):
    v = vector * 3 * np.sqrt(length)  # Scale vector by 3 times the sqrt of its eigenvalue for visibility
    plt.quiver(center_point[0], center_point[1], v[0], v[1], angles='xy', scale_units='xy', scale=1, color='r')

plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('PCA on Points')
plt.axis('equal')
plt.grid(True)
plt.show()

# Print PCA components and explained variance
print("Principal components:\n", pca.components_)
print("Explained variance ratio:\n", pca.explained_variance_ratio_)
