import ConfigParser

configParser = ConfigParser.RawConfigParser()   
configFilePath = r'c:\abc.txt'
configParser.read(configFilePath)
 
self.path = configParser.get('your-config', 'path1')\


[your-config]
path1 = "D:\test1\first"
path2 = "D:\test2\second"
path3 = "D:\test2\third"