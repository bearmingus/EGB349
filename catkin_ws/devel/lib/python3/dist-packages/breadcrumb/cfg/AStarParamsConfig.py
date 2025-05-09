## *********************************************************
##
## File autogenerated for the breadcrumb package
## by the dynamic_reconfigure package.
## Please do not edit.
##
## ********************************************************/

from dynamic_reconfigure.encoding import extract_params

inf = float('inf')

config_description = {'name': 'Default', 'type': '', 'state': True, 'cstate': 'true', 'id': 0, 'parent': 0, 'parameters': [{'name': 'allow_diagonals', 'type': 'bool', 'default': True, 'level': 0, 'description': 'Allows the path tracking to output diagonal movements', 'min': False, 'max': True, 'srcline': 291, 'srcfile': '/opt/ros/noetic/lib/python3/dist-packages/dynamic_reconfigure/parameter_generator_catkin.py', 'edit_method': '', 'ctype': 'bool', 'cconsttype': 'const bool'}, {'name': 'obstacle_threshold', 'type': 'int', 'default': 50, 'level': 0, 'description': 'At what occupancy probability where an obstacle is counted in the search', 'min': 0, 'max': 100, 'srcline': 291, 'srcfile': '/opt/ros/noetic/lib/python3/dist-packages/dynamic_reconfigure/parameter_generator_catkin.py', 'edit_method': '', 'ctype': 'int', 'cconsttype': 'const int'}, {'name': 'search_heuristic', 'type': 'int', 'default': 1, 'level': 0, 'description': 'Heuristic to use during A* search', 'min': 0, 'max': 2, 'srcline': 291, 'srcfile': '/opt/ros/noetic/lib/python3/dist-packages/dynamic_reconfigure/parameter_generator_catkin.py', 'edit_method': "{'enum': [{'name': 'Manhattan', 'type': 'int', 'value': 0, 'srcline': 9, 'srcfile': '/home/cdrone/catkin_ws/src/breadcrumb/cfg/AStarParams.cfg', 'description': 'Sets A* to use the Manhattan heuristic', 'ctype': 'int', 'cconsttype': 'const int'}, {'name': 'Euclidean', 'type': 'int', 'value': 1, 'srcline': 10, 'srcfile': '/home/cdrone/catkin_ws/src/breadcrumb/cfg/AStarParams.cfg', 'description': 'Sets A* to use the Euclidean heuristic', 'ctype': 'int', 'cconsttype': 'const int'}, {'name': 'Octagonal', 'type': 'int', 'value': 2, 'srcline': 11, 'srcfile': '/home/cdrone/catkin_ws/src/breadcrumb/cfg/AStarParams.cfg', 'description': 'Sets A* to use the Octagonal heuristic', 'ctype': 'int', 'cconsttype': 'const int'}], 'enum_description': 'An enum representing the search heuristic'}", 'ctype': 'int', 'cconsttype': 'const int'}, {'name': 'calc_sparse_path', 'type': 'bool', 'default': True, 'level': 0, 'description': 'Enables calculation of the sparse path result', 'min': False, 'max': True, 'srcline': 291, 'srcfile': '/opt/ros/noetic/lib/python3/dist-packages/dynamic_reconfigure/parameter_generator_catkin.py', 'edit_method': '', 'ctype': 'bool', 'cconsttype': 'const bool'}], 'groups': [], 'srcline': 246, 'srcfile': '/opt/ros/noetic/lib/python3/dist-packages/dynamic_reconfigure/parameter_generator_catkin.py', 'class': 'DEFAULT', 'parentclass': '', 'parentname': 'Default', 'field': 'default', 'upper': 'DEFAULT', 'lower': 'groups'}

min = {}
max = {}
defaults = {}
level = {}
type = {}
all_level = 0

#def extract_params(config):
#    params = []
#    params.extend(config['parameters'])
#    for group in config['groups']:
#        params.extend(extract_params(group))
#    return params

for param in extract_params(config_description):
    min[param['name']] = param['min']
    max[param['name']] = param['max']
    defaults[param['name']] = param['default']
    level[param['name']] = param['level']
    type[param['name']] = param['type']
    all_level = all_level | param['level']

AStarParams_Manhattan = 0
AStarParams_Euclidean = 1
AStarParams_Octagonal = 2
