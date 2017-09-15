import configparser
config = configparser.ConfigParser()
config.read('conf.ini')

config.get('file-path','path1')
