import string
import os

def PathCorrection(inData):
	outData=inData
	for i in string.ascii_lowercase: #Replace drive letters with /mnt/
		outData=outData.replace(i+':','/mnt/'+i) #if drive letter is supplied in lowercase
		outData=outData.replace(i.upper()+':','/mnt/'+i) #if drive letter is supplied as uppercase
	outData=outData.replace(os.path.sep, '/') #Change windows filesep to linux filesep

	return outData
