import numpy as np
import os
def read_dict(filename):
 num_keys = 0
 result = {}
 with open(filename) as fp:
  cnt=0
  for line in fp:
   line.strip()
   if cnt == 0:
    pos = line.find("NumEntries:")+11
    num_keys = int(line[pos:])
   else:
    pos0 = line.find("Key:")+5
    pos1 = line.find(",",pos0)
    key = line[pos0:pos1]
    pos0 = line.find("Shape:")+8
    pos1 = line.find("),",pos0)
    shape = list(map(int, line[pos0:pos1].split(', ')))
    ndim = len(shape)
    pos0 = line.find("DataType:")+10
    pos1 = line.find(",",pos0)
    datatype = line[pos0:pos1]
    ndim = len(shape)
    pos = line.find("Data: ")+6
    line2 = line[pos:].replace("["," ").replace("]"," ").replace(","," ").strip()
    if datatype in ["double", "float"]:
     data = list(map(float, line2.split()))
    elif datatype in ["long long int", "long int", "int"]:
     data = list(map(int, line2.split()))
    elif datatype in ["long long unsigned int", "long unsigned int", "unsigned int", "long long int", "long int", "int"]:
     line2 = line2.replace("18446744073709551615","-1")
     data = list(map(int, line2.split()))
    elif datatype == "string":
     if shape[0]!=1 or len(shape)>1:
      data = line2.split("  ")
     else:
      data = line2
    elif datatype == "bool":
     data = list(map(int, line2.split()))
    else:
     print("datatype unkown,",datatype)
    data = np.array(data).reshape(shape)
    result[key] = data
   cnt+=1
 if num_keys != len(result.keys()):
  print("read_dict: something went wrong")
 return result


UNITS = {
    'MU_to_Msun': 1e10,   # Mass Unit to Solar Masses
    't_to_Gyr': 0.979,    # Internal time to Gigayears
    'V_to_kms': 1.0       # Velocity is usually already in km/s
}

def dimfull_data(filename):
  data = read_dict(filename)
  data_dim = {}
  # print(data.keys())
  data_dim["t"] = data["Time"] * UNITS['t_to_Gyr']
  data_dim["rho"] = data["ProfileDensityDM"] * UNITS['MU_to_Msun']
  data_dim["r"] = data["ProfileEdges"] 
  data_dim["err_rho"] = data["ProfileDensityErrDM"] * UNITS['MU_to_Msun']

  return data_dim

if __name__ == "__main__":
  data=read_dict(fr"C:\Users\lukas\Documents\GitHub_new26\Bachelor-Code\Evaluation\Fischer_Data\results_profiles.txt")
  print(data.keys())