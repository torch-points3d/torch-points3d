LABELS = {
    0: "unlabeled",
    1: "outlier",
    10: "car",
    11: "bicycle",
    13: "bus",
    15: "motorcycle",
    16: "on-rails",
    18: "truck",
    20: "other-vehicle",
    30: "person",
    31: "bicyclist",
    32: "motorcyclist",
    40: "road",
    44: "parking",
    48: "sidewalk",
    49: "other-ground",
    50: "building",
    51: "fence",
    52: "other-structure",
    60: "lane-marking",
    70: "vegetation",
    71: "trunk",
    72: "terrain",
    80: "pole",
    81: "traffic-sign",
    99: "other-object",
    252: "moving-car",
    253: "moving-bicyclist",
    254: "moving-person",
    255: "moving-motorcyclist",
    256: "moving-on-rails",
    257: "moving-bus",
    258: "moving-truck",
    259: "moving-other-vehicle",
}

COLOR_MAP = {
    0: [0, 0, 0],
    1: [0, 0, 255],
    10: [245, 150, 100],
    11: [245, 230, 100],
    13: [250, 80, 100],
    15: [150, 60, 30],
    16: [255, 0, 0],
    18: [180, 30, 80],
    20: [255, 0, 0],
    30: [30, 30, 255],
    31: [200, 40, 255],
    32: [90, 30, 150],
    40: [255, 0, 255],
    44: [255, 150, 255],
    48: [75, 0, 75],
    49: [75, 0, 175],
    50: [0, 200, 255],
    51: [50, 120, 255],
    52: [0, 150, 255],
    60: [170, 255, 150],
    70: [0, 175, 0],
    71: [0, 60, 135],
    72: [80, 240, 150],
    80: [150, 240, 255],
    81: [0, 0, 255],
    99: [255, 255, 50],
    252: [245, 150, 100],
    256: [255, 0, 0],
    253: [200, 40, 255],
    254: [30, 30, 255],
    255: [90, 30, 150],
    257: [250, 80, 100],
    258: [180, 30, 80],
    259: [255, 0, 0],
}

# ratio of points in a class with respect to the total number
CONTENT = {
    0: 0.018889854628292943,
    1: 0.0002937197336781505,
    10: 0.040818519255974316,
    11: 0.00016609538710764618,
    13: 2.7879693665067774e-05,
    15: 0.00039838616015114444,
    16: 0.0,
    18: 0.0020633612104619787,
    20: 0.0016218197275284021,
    30: 0.00017698551338515307,
    31: 1.1065903904919655e-08,
    32: 5.532951952459828e-09,
    40: 0.1987493871255525,
    44: 0.014717169549888214,
    48: 0.14392298360372,
    49: 0.0039048553037472045,
    50: 0.1326861944777486,
    51: 0.0723592229456223,
    52: 0.002395131480328884,
    60: 4.7084144280367186e-05,
    70: 0.26681502148037506,
    71: 0.006035012012626033,
    72: 0.07814222006271769,
    80: 0.002855498193863172,
    81: 0.0006155958086189918,
    99: 0.009923127583046915,
    252: 0.001789309418528068,
    253: 0.00012709999297008662,
    254: 0.00016059776092534436,
    255: 3.745553104802113e-05,
    256: 0.0,
    257: 0.00011351574470342043,
    258: 0.00010157861367183268,
    259: 4.3840131989471124e-05,
}

# objects which are not identifiable from a single scan are mapped to their closest
REMAPPING_MAP = {
    0: -1,  # "unlabeled"
    1: -1,  # "outlier" mapped to "unlabeled" --------------------------mapped
    10: 0,  # "car"
    11: 1,  # "bicycle"
    13: 4,  # "bus" mapped to "other-vehicle" --------------------------mapped
    15: 2,  # "motorcycle"
    16: 4,  # "on-rails" mapped to "other-vehicle" ---------------------mapped
    18: 3,  # "truck"
    20: 4,  # "other-vehicle"
    30: 5,  # "person"
    31: 6,  # "bicyclist"
    32: 7,  # "motorcyclist"
    40: 8,  # "road"
    44: 9,  # "parking"
    48: 10,  # "sidewalk"
    49: 11,  # "other-ground"
    50: 12,  # "building"
    51: 13,  # "fence"
    52: -1,  # "other-structure" mapped to "unlabeled" ------------------mapped
    60: 8,  # "lane-marking" to "road" ---------------------------------mapped
    70: 14,  # "vegetation"
    71: 15,  # "trunk"
    72: 16,  # "terrain"
    80: 17,  # "pole"
    81: 18,  # "traffic-sign"
    99: -1,  # "other-object" to "unlabeled" ----------------------------mapped
    252: 0,  # "moving-car" to "car" ------------------------------------mapped
    253: 6,  # "moving-bicyclist" to "bicyclist" ------------------------mapped
    254: 5,  # "moving-person" to "person" ------------------------------mapped
    255: 7,  # "moving-motorcyclist" to "motorcyclist" ------------------mapped
    256: 4,  # "moving-on-rails" mapped to "other-vehicle" --------------mapped
    257: 4,  # "moving-bus" mapped to "other-vehicle" -------------------mapped
    258: 3,  # "moving-truck" to "truck" --------------------------------mapped
    259: 4,  # "moving-other"-vehicle to "other-vehicle" ----------------mapped
}

# invert above feature map
LEARNING_MAP_INV = {
    -1: 0,  # "unlabeled", and others ignored
    0: 10,  # "car"
    1: 11,  # "bicycle"
    2: 15,  # "motorcycle"
    3: 18,  # "truck"
    4: 20,  # "other-vehicle"
    5: 30,  # "person"
    6: 31,  # "bicyclist"
    7: 32,  # "motorcyclist"
    8: 40,  # "road"
    9: 44,  # "parking"
    10: 48,  # "sidewalk"
    11: 49,  # "other-ground"
    12: 50,  # "building"
    13: 51,  # "fence"
    14: 70,  # "vegetation"
    15: 71,  # "trunk"
    16: 72,  # "terrain"
    17: 80,  # "pole"
    18: 81,  # "traffic-sign"
}

# sequences in split types
SPLIT = {"train": [0, 1, 2, 3, 4, 5, 6, 7, 9, 10], "val": [8], "test": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]}

# sensor configuration
SENSOR_CONFIG = {"name": "HDL64", "type": "spherical", "fov_up": 3, "fov_down": -25}

# projected image properties
IMG_PROP = {
    # range, x, y, z signal
    "img_means": [12.12, 10.88, 0.23, -1.04, 0.21],
    "img_stds": [12.32, 11, 47, 6, 91, 0.86, 0.16],
}

