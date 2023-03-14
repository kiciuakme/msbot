from utils import *

# for cd in LocalMask([[1,1,1,1,0],[0,0,0,0,1],[0,0,0,0,1],[1,1,0,0,0],[0,0,1,1,0]]).relative_coordinates_generator():
#     print(cd.x, cd.y)

for m in AdjacentMask([[1,1,1],[0,0,0],[0,0,0]]).get_coordinate_negation(Cartesian2d(1, 1)).submask_generator(1):
    print(m.mask)

# print(LocalMask(True).get_exclude(AdjacentMask(True), Cartesian2d(3,1)).mask)
