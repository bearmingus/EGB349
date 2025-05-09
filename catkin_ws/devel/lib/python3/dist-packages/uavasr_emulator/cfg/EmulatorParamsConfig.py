## *********************************************************
##
## File autogenerated for the uavasr_emulator package
## by the dynamic_reconfigure package.
## Please do not edit.
##
## ********************************************************/

from dynamic_reconfigure.encoding import extract_params

inf = float('inf')

config_description = {'name': 'Default', 'type': '', 'state': True, 'cstate': 'true', 'id': 0, 'parent': 0, 'parameters': [], 'groups': [{'name': 'Autopilot', 'type': '', 'state': True, 'cstate': 'true', 'id': 1, 'parent': 0, 'parameters': [{'name': 'auto_disarm_height', 'type': 'double', 'default': 0.2, 'level': 0, 'description': 'Height to enable automatic disarm (set to 0 to disable)', 'min': 0.0, 'max': inf, 'srcline': 29, 'srcfile': '/home/cdrone/catkin_ws/src/uavasr_emulator/cfg/EmulatorParams.cfg', 'edit_method': '', 'ctype': 'double', 'cconsttype': 'const double'}], 'groups': [], 'srcline': 124, 'srcfile': '/opt/ros/noetic/lib/python3/dist-packages/dynamic_reconfigure/parameter_generator_catkin.py', 'class': 'DEFAULT::AUTOPILOT', 'parentclass': 'DEFAULT', 'parentname': 'Default', 'field': 'DEFAULT::autopilot', 'upper': 'AUTOPILOT', 'lower': 'autopilot'}, {'name': 'Control', 'type': '', 'state': True, 'cstate': 'true', 'id': 2, 'parent': 0, 'parameters': [{'name': 'w0_xy', 'type': 'double', 'default': 1.0, 'level': 0, 'description': 'Natural frequency for lateral position', 'min': 0.0, 'max': inf, 'srcline': 32, 'srcfile': '/home/cdrone/catkin_ws/src/uavasr_emulator/cfg/EmulatorParams.cfg', 'edit_method': '', 'ctype': 'double', 'cconsttype': 'const double'}, {'name': 'w0_z', 'type': 'double', 'default': 1.0, 'level': 0, 'description': 'Natural frequency vertical position', 'min': 0.0, 'max': inf, 'srcline': 33, 'srcfile': '/home/cdrone/catkin_ws/src/uavasr_emulator/cfg/EmulatorParams.cfg', 'edit_method': '', 'ctype': 'double', 'cconsttype': 'const double'}, {'name': 'w0_psi', 'type': 'double', 'default': 1.0, 'level': 0, 'description': 'Natural frequency for the heading', 'min': 0.0, 'max': inf, 'srcline': 34, 'srcfile': '/home/cdrone/catkin_ws/src/uavasr_emulator/cfg/EmulatorParams.cfg', 'edit_method': '', 'ctype': 'double', 'cconsttype': 'const double'}, {'name': 'max_vel_xy', 'type': 'double', 'default': 2.5, 'level': 0, 'description': 'Maximum horizontal velocity', 'min': 0.0, 'max': inf, 'srcline': 35, 'srcfile': '/home/cdrone/catkin_ws/src/uavasr_emulator/cfg/EmulatorParams.cfg', 'edit_method': '', 'ctype': 'double', 'cconsttype': 'const double'}, {'name': 'max_vel_z', 'type': 'double', 'default': 1.0, 'level': 0, 'description': 'Natural vertical velocity', 'min': 0.0, 'max': inf, 'srcline': 36, 'srcfile': '/home/cdrone/catkin_ws/src/uavasr_emulator/cfg/EmulatorParams.cfg', 'edit_method': '', 'ctype': 'double', 'cconsttype': 'const double'}], 'groups': [], 'srcline': 124, 'srcfile': '/opt/ros/noetic/lib/python3/dist-packages/dynamic_reconfigure/parameter_generator_catkin.py', 'class': 'DEFAULT::CONTROL', 'parentclass': 'DEFAULT', 'parentname': 'Default', 'field': 'DEFAULT::control', 'upper': 'CONTROL', 'lower': 'control'}, {'name': 'System', 'type': '', 'state': True, 'cstate': 'true', 'id': 3, 'parent': 0, 'parameters': [{'name': 'rate_state', 'type': 'double', 'default': 1.0, 'level': 0, 'description': 'Update rate for system state', 'min': 0.0, 'max': inf, 'srcline': 39, 'srcfile': '/home/cdrone/catkin_ws/src/uavasr_emulator/cfg/EmulatorParams.cfg', 'edit_method': '', 'ctype': 'double', 'cconsttype': 'const double'}, {'name': 'rate_battery', 'type': 'double', 'default': 1.0, 'level': 0, 'description': 'Update rate for battery state', 'min': 0.0, 'max': inf, 'srcline': 40, 'srcfile': '/home/cdrone/catkin_ws/src/uavasr_emulator/cfg/EmulatorParams.cfg', 'edit_method': '', 'ctype': 'double', 'cconsttype': 'const double'}, {'name': 'rate_pose', 'type': 'double', 'default': 50.0, 'level': 0, 'description': 'Update rate for system pose', 'min': 0.0, 'max': inf, 'srcline': 41, 'srcfile': '/home/cdrone/catkin_ws/src/uavasr_emulator/cfg/EmulatorParams.cfg', 'edit_method': '', 'ctype': 'double', 'cconsttype': 'const double'}], 'groups': [], 'srcline': 124, 'srcfile': '/opt/ros/noetic/lib/python3/dist-packages/dynamic_reconfigure/parameter_generator_catkin.py', 'class': 'DEFAULT::SYSTEM', 'parentclass': 'DEFAULT', 'parentname': 'Default', 'field': 'DEFAULT::system', 'upper': 'SYSTEM', 'lower': 'system'}], 'srcline': 246, 'srcfile': '/opt/ros/noetic/lib/python3/dist-packages/dynamic_reconfigure/parameter_generator_catkin.py', 'class': 'DEFAULT', 'parentclass': '', 'parentname': 'Default', 'field': 'default', 'upper': 'DEFAULT', 'lower': 'groups'}

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

