# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 17:46:27 2012

@author: pedro.correia
"""

from __future__ import division
#import sys
#reload(sys)  # Reload does the trick!
#sys.setdefaultencoding('UTF8')
import numpy as np
import os
from struct import unpack

"""
This is the first module for CERENA library with several functions that depend
only on numpy and native Python. It was written in Python 2.7.2.
"""

"""
LINK: http://docs.scipy.org/doc/numpy/user/basics.types.html
Data type 	Description
bool 	Boolean (True or False) stored as a byte
int 	Platform integer (normally either int32 or int64)
int8 	Byte (-128 to 127)
int16 	Integer (-32768 to 32767)
int32 	Integer (-2147483648 to 2147483647)
int64 	Integer (9223372036854775808 to 9223372036854775807)
uint8 	Unsigned integer (0 to 255)
uint16 	Unsigned integer (0 to 65535)
uint32 	Unsigned integer (0 to 4294967295)
uint64 	Unsigned integer (0 to 18446744073709551615)
float 	Shorthand for float64.
float16 	Half precision float: sign bit, 5 bits exponent, 10 bits mantissa
float32 	Single precision float: sign bit, 8 bits exponent, 23 bits mantissa
float64 	Double precision float: sign bit, 11 bits exponent, 52 bits mantissa
complex 	Shorthand for complex128.
complex64 	Complex number, represented by two 32-bit floats (real and imaginary components)
complex128 	Complex number, represented by two 64-bit floats (real and imaginary components)
"""
 
def BinarySTL(fname):
    fp = open(fname, 'rb')
    Header = fp.read(80)
    nn = fp.read(4)
    Numtri = unpack('i', nn)[0]
    #print nn
    record_dtype = np.dtype([
                   ('normals', np.float32,(3,)), 
                   ('Vertex1', np.float32,(3,)),
                   ('Vertex2', np.float32,(3,)),
                   ('Vertex3', np.float32,(3,)) ,             
                   ('atttr', '<i2',(1,) )
    ])
    data = np.fromfile(fp , dtype = record_dtype , count =Numtri)
    fp.close()
 
    #Normals = data['normals']
    Vertex1= data['Vertex1']
    Vertex2= data['Vertex2']
    Vertex3= data['Vertex3']
    x = np.hstack((Vertex1[:,0][:,np.newaxis],Vertex2[:,0][:,np.newaxis],Vertex3[:,0][:,np.newaxis])).flatten()
    y = np.hstack((Vertex1[:,1][:,np.newaxis],Vertex2[:,1][:,np.newaxis],Vertex3[:,1][:,np.newaxis])).flatten()
    z = np.hstack((Vertex1[:,2][:,np.newaxis],Vertex2[:,2][:,np.newaxis],Vertex3[:,2][:,np.newaxis])).flatten()
    """
    full = Vertex1.shape[0]+Vertex2.shape[0]+Vertex3.shape[0]
    x=np.zeros(full,dtype='int32')
    y=np.zeros(full,dtype='int32')
    z=np.zeros(full,dtype='int32')
    t=0
    c=0
    while t<x.shape[0]:
        x[t] = Vertex1[c,0]
        x[t+1] = Vertex2[c,0]
        x[x+2] = Vertex3[c,0]
        y[t] = Vertex1[c,1]
        y[t+1] = Vertex2[c,1]
        y[x+2] = Vertex3[c,1]
        z[t] = Vertex1[c,2]
        z[t+1] = Vertex2[c,2]
        z[x+2] = Vertex3[c,2]
        t=t+3
        c=c+1
    """
    #x=np.hstack((Vertex1[:,0],Vertex2[:,0],Vertex3[:,0]))
    #y=Vertex2.flatten() #np.hstack((Vertex1[:,1],Vertex2[:,1],Vertex3[:,1]))
    #z=Vertex3.flatten() #np.hstack((Vertex1[:,2],Vertex2[:,2],Vertex3[:,2]))
    #print Vertex1.shape
    #p = np.append(Vertex1,Vertex2,axis=0)
    #p = np.append(p,Vertex3,axis=0) #list(v1)
    #Points =np.array(list(set(tuple(p1) for p1 in p)))
    #print Points.shape
    #print Points[0:3,:]
    #print Normals.shape
    triangles = np.zeros((int(x.shape[0]/3),3),dtype='int32')
    c=0
    t=0
    while c<triangles.shape[0]:
        triangles[c,:] = np.array([t,t+1,t+2])[:]
        c=c+1
        t=t+3
    return Header,x,y,z,triangles

def __manage_directory__(path):
    '''
    __manage_directory__(...)
        __manage_directory__(path)
        
        Checks if directory exists. If it does not than its created.
        
    Parameters
    ----------
    path : string
        String with directory path.
        
    Returns
    -------
    out: None
    
    See also
    --------
    __manage_file_numbering__
    '''
    d = os.path.dirname(path)
    if not os.path.exists(d):
        os.makedirs(d)
        
def __manage_file_numbering__(directory,name):
    '''
    __manage_file_numbering__(...)
        __manage_file_numbering__(path,name)
        
        Checks with what name a file can be created in the given directory
        path (by including a random number and extension .DAT).
        
    Parameters
    ----------
    path : string
        String with directory path.
        
    name : string
        Name of a given file (to be transformed with a number and extension
        .DAT)
        
    Returns
    -------
    out: string
        String with file path that can be safely created (path includes
        directory).
    
    See also
    --------
    __manage_directory__
    '''
    directory_list_of_files = os.listdir(directory)
    chosen_name = name+'_'+str(np.random.randint(1000,9999))+'.dat'
    while chosen_name in directory_list_of_files:
        chosen_name = name+'_'+str(np.random.randint(1000,9999))+'.dat'
    return directory+'\\'+chosen_name
        
def check_header_on_file(path,at_least = 3):
    '''
    check_header_on_file(...)
        check_header_on_file(path,at_least)
        
        Calculates the number of rows in a file which are not a digit. Uses
        argument at_least to check how many lines in a row are digit. Once
        it gets to at_least number than it calculates the number of header rows.
        
    Parameters
    ----------
    path : string
        String with file path.
        
    at_least : int
        Integer giving the number of lines in row which are digit. If it gets
        to this number than its assumed the information is no longer header
        information.
        
    Returns
    -------
    out: int
        Integer with the number size of header rows.
    
    See also
    --------
    check_header_special_char_on_file,check_number_of_columns,
    check_number_of_columns_special_char,check_dtype_on_file
    '''
    fid = open(path,'r')
    flag = True
    counter = at_least
    flag = False
    cc = 0
    while counter != 0:
        if fid.readline().replace('\n','').replace('.','').replace('-','').replace('e','').replace('+','').split()[0].isdigit():
            counter = counter - 1
            cc = cc + 1
            flag = True
        else:
            if flag:
                counter = at_least
                flag = False
                cc = cc + 1
            else:
                cc = cc + 1
    fid.close()
    return cc-at_least
    
def check_header_on_flexible_file(path,at_least = 3):
    '''
    check_header_on_flexible_file(...)
        check_header_on_flexible_file(path,at_least)
        
        Calculates the number of rows in a file which are not a digit. Uses
        argument at_least to check how many lines in a row are digit. Once
        it gets to at_least number than it calculates the number of header rows.
        For the flexible case it must have at_least rows with one column being
        numeric for the header to be considered finished.
        
    Parameters
    ----------
    path : string
        String with file path.
        
    at_least : int
        Integer giving the number of lines in row which are digit. If it gets
        to this number than its assumed the information is no longer header
        information.
        
    Returns
    -------
    out: int
        Integer with the number size of header rows.
    
    See also
    --------
    check_header_special_char_on_file,check_number_of_columns,
    check_number_of_columns_special_char,check_dtype_on_file
    '''
    fid = open(path,'r')
    flag = True
    counter = at_least
    flag = False
    cc = 0
    while counter != 0:
        line = fid.readline().replace('\n','').replace('.','').replace('-','').replace('e','').replace('+','').split()
        lflag = False
        for i in line:
            if i.isdigit(): lflag = True
        if lflag:
            counter = counter - 1
            cc = cc + 1
            flag = True
        else:
            if flag:
                counter = at_least
                flag = False
                cc = cc + 1
            else:
                cc = cc + 1
    fid.close()
    return cc-at_least
    
    
def check_header_special_char_on_file(path,at_least = 3):
    '''
    check_header_special_char_on_file(...)
        check_header_special_char_on_file(path,at_least)
        
        Special files where data information is stored in columns separated
        by ; or , or _ instead of whitespace.
        
        Calculates the number of rows in a file which are not a digit. Uses
        argument at_least to check how many lines in a row are digit. Once
        it gets to at_least number than it calculates the number of header rows.
        
    Parameters
    ----------
    path : string
        String with file path.
        
    at_least : int
        Integer giving the number of lines in row which are digit. If it gets
        to this number than its assumed the information is no longer header
        information.
        
    Returns
    -------
    out: tuple
        Tuple (Integer with the number size of header rows, separator char)
    
    See also
    --------
    check_header_on_file,check_number_of_columns,
    check_number_of_columns_special_char,check_dtype_on_file
    '''
    fid = open(path,'r')
    flag = True
    counter = at_least
    flag = False
    cc = 0
    while counter != 0:
        appex =  fid.readline().replace('\n','')
        if appex.replace('.','').split()[0].isdigit():
            counter = counter - 1
            cc = cc + 1
            char = ' '
            flag = True
        elif appex.replace('.','').split(';')[0].isdigit():
            counter = counter - 1
            cc = cc + 1
            char = ';'
            flag = True
        elif appex.replace('.','').split(',')[0].isdigit():
            counter = counter - 1
            cc = cc + 1
            char = ','
            flag = True
        elif appex.replace('.','').split('_')[0].isdigit():
            counter = counter - 1
            cc = cc + 1
            char = '_'
            flag = True
        else:
            if flag:
                counter = at_least
                flag = False
                cc = cc + 1
            else:
                cc = cc + 1
    fid.close()
    return (cc-at_least,char)
    
def check_number_of_columns(path,header):
    '''
    check_number_of_columns(...)
        check_number_of_columns(path,header)
        
        Calculates the number of columns the data information has.
        
    Parameters
    ----------
    path : string
        String with file path.
        
    header : int
        Integer with the size of header rows.
        
    Returns
    -------
    out: int
        Integer with the number of columns of data information in file.
    
    See also
    --------
    check_header_special_char_on_file,check_header_on_file,
    check_number_of_columns_special_char,check_dtype_on_file
    '''
    fid = open(path,'r')
    for i in xrange(header): fid.readline()
    columns = len(fid.readline().split())
    return columns
    
def check_number_of_columns_special_char(path,header,char):
    '''
    check_number_of_columns_special_char(...)
        check_number_of_columns_special_char(path,header,char)
        
        Calculates the number of columns the data information has considering
        a specific data (or field) separator.
        
    Parameters
    ----------
    path : string
        String with file path.
        
    header : int
        Integer with the size of header rows.
        
    char : string
        String character to be used as separator.
        
    Returns
    -------
    out: int
        Integer with the number of columns of data information in file.
    
    See also
    --------
    check_header_special_char_on_file,check_header_on_file,
    check_number_of_columns,check_dtype_on_file
    '''
    fid = open(path,'r')
    for i in xrange(header): fid.readline()
    columns = len(fid.readline().split(char))
    return columns
    
def check_dtype_on_file(path,header,columns):
    '''
    check_dtype_on_file(...)
        check_dtype_on_file(path,header,columns)
        
        Attempts to check if information in file is integer or float.
        Its assumed its reading numeric information on file.
        
    Parameters
    ----------
    path : string
        String with file path.
        
    header : int
        Integer with the size of header rows.
        
    columns : int
        Integer with the size of data columns.
        
    Returns
    -------
    out: string
        String with the result of the test. Either int or float32.
    
    See also
    --------
    check_header_special_char_on_file,check_header_on_file,
    check_number_of_columns,check_dtype_on_file
    '''
    fid = open(path,'r')
    for i in xrange(header): fid.readline()
    if columns == 1:
        line = fid.readline().replace('\n','')
        if '.' in line:
            return 'float32'
        else:
            return 'int'
    else:
        line = fid.readline().replace('\n','').split()
        for i in xrange(len(line)):
            if '.' in line[i]:
                return 'float32'
        return 'int'
        
def swap_load_ascii_single_grid(path,blocks,header=0,dtype='float32',swap_directory='TMP'):
    '''
    swap_load_ascii_single_grid(...)
        swap_load_ascii_single_grid(path,blocks,header,dtype,swap_directory)
        
        Loads a single (one variable) grid (mesh) ASCII file into a memory
        swap space.
        
    Parameters
    ----------
    path : string
        String with file path.
        
    blocks : tuple
        Tuple with number of blocks in (X,Y,Z).
        
    header : int
        Integer with the size of header rows.
        
    dtype : string
        Memory swap space data type. Default is float32.
        
    swap_directory : string
        String with directory to build the swap file.
        
    Returns
    -------
    out: swap array
        3D numpy swap array with information on file.
    
    See also
    --------
    swap_load_ascii_multiple_grid,load_ascii_single_grid,load_ascii_multiple_grid
    swap_load_npy_grid,load_ascii_grid,load_npy_grid,load_grid
    '''
    fid = open(path,'r')
    for i in xrange(header): fid.readline()
    directory_list_of_files = os.listdir(swap_directory)
    chosen_name = 'grid_'+str(np.random.randint(1000,9999))+'.dat'
    while chosen_name in directory_list_of_files:
        chosen_name = 'grid_'+str(np.random.randint(1000,9999))+'.dat'
    mem_grid = np.memmap(swap_directory+'\\'+chosen_name, dtype=dtype, mode='w+', shape=blocks, order='F')
    if dtype in ['float','float16','float32','float64']:
        for z in xrange(blocks[2]):
            for y in xrange(blocks[1]):
                for x in xrange(blocks[0]):
                    mem_grid[x,y,z] = np.float(fid.readline())
    elif dtype == 'bool':
        for z in xrange(blocks[2]):
            for y in xrange(blocks[1]):
                for x in xrange(blocks[0]):
                    mem_grid[x,y,z] = np.bool(np.int(fid.readline()))
    elif dtype in ['int','int8','int16','int32','int64','uint8','uint16','uint32','uint64']:
        for z in xrange(blocks[2]):
            for y in xrange(blocks[1]):
                for x in xrange(blocks[0]):
                    mem_grid[x,y,z] = np.int(fid.readline())
    fid.close()
    return mem_grid
    
def swap_load_ascii_multiple_grid(path,blocks,header=0,dtype='float32',columns = 2,swap_directory='TMP'):
    '''
    swap_load_ascii_multiple_grid(...)
        swap_load_ascii_multiple_grid(path,blocks,header,dtype,columns,swap_directory)
        
        Loads a multiple (multiple columns) grid (mesh) ASCII file into a memory
        swap space.
        
    Parameters
    ----------
    path : string
        String with file path.
        
    blocks : tuple
        Tuple with number of blocks in (X,Y,Z).
        
    header : int
        Integer with the size of header rows.
        
    dtype : string
        Memory swap space data type. Default is float32.
        
    columns : int
        Integer with the size of data columns. Default is 2.
        
    swap_directory : string
        String with directory to build the swap file.
        
    Returns
    -------
    out: swap array
        4D numpy swap array with information on file (4th dimension is given by
        number of variables).
    
    See also
    --------
    swap_load_ascii_single_grid,load_ascii_single_grid,load_ascii_multiple_grid
    swap_load_npy_grid,load_ascii_grid,load_npy_grid,load_grid
    '''
    fid = open(path,'r')
    for i in xrange(header): fid.readline()
    directory_list_of_files = os.listdir(swap_directory)
    chosen_name = 'grid_'+str(np.random.randint(1000,9999))+'.dat'
    while chosen_name in directory_list_of_files:
        chosen_name = 'grid_'+str(np.random.randint(1000,9999))+'.dat'
    mem_grid = np.memmap(swap_directory+'\\'+chosen_name, dtype=dtype, mode='w+', shape=(blocks[0],blocks[1],blocks[2],columns), order='F')
    if dtype in ['float','float16','float32','float64']:
        for z in xrange(blocks[2]):
            for y in xrange(blocks[1]):
                for x in xrange(blocks[0]):
                    mem_grid[x,y,z,:] = np.float_(fid.readline().split())
    elif dtype == 'bool':
        for z in xrange(blocks[2]):
            for y in xrange(blocks[1]):
                for x in xrange(blocks[0]):
                    mem_grid[x,y,z,:] = np.bool_(np.int_(fid.readline().split()))
    elif dtype in ['int','int8','int16','int32','int64','uint8','uint16','uint32','uint64']:
        for z in xrange(blocks[2]):
            for y in xrange(blocks[1]):
                for x in xrange(blocks[0]):
                    mem_grid[x,y,z,:] = np.int_(fid.readline().split())
    fid.close()
    return mem_grid
    
    
def load_ascii_single_grid(path,blocks,header=0,dtype='float32'):
    '''
    load_ascii_single_grid(...)
        load_ascii_single_grid(path,blocks,header,dtype)
        
        Loads a single (one column) grid (mesh) ASCII file.
        
    Parameters
    ----------
    path : string
        String with file path.
        
    blocks : tuple
        Tuple with number of blocks in (X,Y,Z).
        
    header : int
        Integer with the size of header rows.
        
    dtype : string
        String with data type. Default is float32.
        
        
    Returns
    -------
    out: numpy array
        3D numpy array with information on file.
    
    See also
    --------
    swap_load_ascii_single_grid,swap_load_ascii_nultiple_grid,load_ascii_multiple_grid
    swap_load_npy_grid,load_ascii_grid,load_npy_grid,load_grid
    '''
    fid = open(path,'r')
    for i in xrange(header): fid.readline()
    grid = np.zeros(blocks,dtype=dtype,order='F')
    if dtype in ['float','float16','float32','float64']:
        for z in xrange(blocks[2]):
            for y in xrange(blocks[1]):
                for x in xrange(blocks[0]):
                    grid[x,y,z] = np.float(fid.readline())
    elif dtype == 'bool':
        for z in xrange(blocks[2]):
            for y in xrange(blocks[1]):
                for x in xrange(blocks[0]):
                    grid[x,y,z] = np.bool(np.int(fid.readline()))
    elif dtype in ['int','int8','int16','int32','int64','uint8','uint16','uint32','uint64']:
        for z in xrange(blocks[2]):
            for y in xrange(blocks[1]):
                for x in xrange(blocks[0]):
                    grid[x,y,z] = np.int(fid.readline())
    fid.close()
    return grid
    
def load_ascii_multiple_grid(path,blocks,header=0,dtype='float32',columns=2):
    '''
    load_ascii_nultiple_grid(...)
        load_ascii_nultiple_grid(path,blocks,header,dtype,columns)
        
        Loads a multiple (multiple columns) grid (mesh) ASCII file.
        
    Parameters
    ----------
    path : string
        String with file path.
        
    blocks : tuple
        Tuple with number of blocks in (X,Y,Z).
        
    header : int
        Integer with the size of header rows.
        
    dtype : string
        String with data type. Default is float32.
        
    columns : int
        Integer with the size of data columns. Default is 2.        
        
    Returns
    -------
    out: numpy array
        4D numpy array with information on file (4th dimension is given by
        number of variables).
    
    See also
    --------
    swap_load_ascii_single_grid,swap_load_ascii_nultiple_grid,load_ascii_single_grid
    swap_load_npy_grid,load_ascii_grid,load_npy_grid,load_grid
    '''
    fid = open(path,'r')
    for i in xrange(header): fid.readline()
    grid = np.zeros((blocks[0],blocks[1],blocks[2],columns),dtype=dtype,order='F')
    if dtype in ['float','float16','float32','float64']:
        for z in xrange(blocks[2]):
            for y in xrange(blocks[1]):
                for x in xrange(blocks[0]):
                    grid[x,y,z,:] = np.float_(fid.readline().split())
    elif dtype == 'bool':
        for z in xrange(blocks[2]):
            for y in xrange(blocks[1]):
                for x in xrange(blocks[0]):
                    grid[x,y,z,:] = np.bool_(np.int_(fid.readline().split()))
    elif dtype in ['int','int8','int16','int32','int64','uint8','uint16','uint32','uint64']:
        for z in xrange(blocks[2]):
            for y in xrange(blocks[1]):
                for x in xrange(blocks[0]):
                    grid[x,y,z,:] = np.int_(fid.readline().split())
    fid.close()
    return grid
    
def swap_load_npy_grid(path,swap_directory='TMP'):
    '''
    swap_load_npy_grid(...)
        swap_load_npy_grid(path,swap_directory)
        
        Loads a grid (mesh, with one or more variable in the 4th dimension)
        NPY (numpy binary) file into memory swap variable.
        
    Parameters
    ----------
    path : string
        String with file path.
        
    swap_directory : string
        String with directory to build the swap file.        
        
    Returns
    -------
    out: numpy swap array
        3D/4D numpy swap array with information on file (4th dimension is given by
        number of variables).
    
    See also
    --------
    swap_load_ascii_single_grid,swap_load_ascii_nultiple_grid,load_ascii_single_grid
    load_ascii_multiple_grid,load_ascii_grid,load_npy_grid,load_grid
    '''
    directory_list_of_files = os.listdir(swap_directory)
    chosen_name = 'grid_'+str(np.random.randint(1000,9999))+'.dat'
    while chosen_name in directory_list_of_files:
        chosen_name = 'grid_'+str(np.random.randint(1000,9999))+'.dat'
    grid = np.load(path)
    mem_grid = np.memmap(swap_directory+'\\'+chosen_name, dtype=grid.dtype, mode='w+', shape=grid.shape, order='F')
    mem_grid[:] = grid[:]
    del(grid)
    return mem_grid
    

def load_ascii_grid(path,blocks,header=0,dtype='float32',columns=1,swap=False,swap_directory='TMP'):
    '''
    load_ascii_grid(...)
        load_ascii_grid(path,blocks,header,dtype,columns,swap,swap_directory)
        
        Load an ASCII grid (mesh) from a file.
        
    Parameters
    ----------
    path : string
        String with file path.
        
    blocks : tuple
        Tuple with number of blocks in (X,Y,Z).
        
    header : int
        Integer with the size of header rows.
        
    dtype : string
        String with data type. Default is float32.
        
    columns : int
        Integer with the size of data columns. Default is 1 (single column).
        
    swap : bool
        Boolean value indicating if output variable should be memory swap.
        Default is False.
        
    swap_directory : string
        String with directory to build the swap file (if swap is True).
        
    Returns
    -------
    out: numpy swap array (or swap)
        3D/4D numpy array with information on file (4th dimension is given by
        number of variables).
    
    See also
    --------
    swap_load_ascii_single_grid,swap_load_ascii_nultiple_grid,load_ascii_single_grid
    load_ascii_multiple_grid,swap_load_npy_grid,load_npy_grid,load_grid
    '''
    if swap:
        __manage_directory__(swap_directory)
        if columns == 1:
            return swap_load_ascii_single_grid(path,blocks,header,dtype,swap_directory)
        else:
            return swap_load_ascii_multiple_grid(path,blocks,header,dtype,columns,swap_directory)
    else:
        if columns == 1:
            return load_ascii_single_grid(path,blocks,header,dtype)
        else:
            return load_ascii_multiple_grid(path,blocks,header,dtype,columns)
            
def load_npy_grid(path,swap=False,swap_directory='TMP'):
    '''
    load_npy_grid(...)
        load_npy_grid(path,swap,swap_directory)
        
        Loads a grid (mesh, with one or more variable in the 4th dimension)
        NPY (numpy binary) file.
        
    Parameters
    ----------
    path : string
        String with file path.
        
    swap : bool
        Boolean value indicating if output variable should be memory swap.
        Default is False.
        
    swap_directory : string
        String with directory to build the swap file (if swap is True).      
        
    Returns
    -------
    out: numpy array (or swap)
        3D/4D numpy array with information on file (4th dimension is given by
        number of variables).
    
    See also
    --------
    swap_load_ascii_single_grid,swap_load_ascii_nultiple_grid,load_ascii_single_grid
    load_ascii_multiple_grid,load_ascii_grid,swap_load_npy_grid,load_grid
    '''
    if swap:
        __manage_directory__(swap_directory)
        return swap_load_npy_grid(path,swap_directory)
    else:
        return np.load(path)
        
def load_grid(path,blocks = (1,1,1),dtype='float32',swap=False,swap_directory='TMP',at_least=3):
    '''
    load_grid(...)
        load_grid(path,blocks,header,dtype,columns,swap,swap_directory)
        
        Loads a grid (mesh) from a file. If file path ends in .npy its assumed
        its a numpy binary (NPY file). Tests to number and columns (whitespace
        separator) are made.
        
    Parameters
    ----------
    path : string
        String with file path.
        
    blocks : tuple
        Tuple with number of blocks in (X,Y,Z).
        
    dtype : string
        String with data type. Default is float32.
        
    swap : bool
        Boolean value indicating if output variable should be memory swap.
        Default is False.
        
    swap_directory : string
        String with directory to build the swap file (if swap is True).
        Default is TMP.
        
    at_least : int
        Integer giving the number of lines in row which are digit. If it gets
        to this number than its assumed the information is no longer header
        information. Default is 3.
        
    Returns
    -------
    out: numpy swap array (or swap)
        3D/4D numpy array with information on file (4th dimension is given by
        number of variables).
    
    See also
    --------
    swap_load_ascii_single_grid,swap_load_ascii_nultiple_grid,load_ascii_single_grid
    load_ascii_multiple_grid,swap_load_npy_grid,load_npy_grid,load_ascii_grid
    '''
    if path[-4:] == '.npy':
        return load_npy_grid(path,swap,swap_directory)
    else:
        if type(blocks) == tuple:
            if len(blocks) == 2:
                blocks = (blocks[0],blocks[1],1)
            elif len(blocks) == 1:
                blocks = (blocks[0],1,1)
            elif len(blocks) == 3:
                pass
            else:
                print 'ERROR ON "load_grid": length of blocks tuple not recognized. Only length 1,2 or 3 accepted.'
                return False
            header = check_header_on_file(path,at_least)
            columns = check_number_of_columns(path,header)
            if type(dtype)==bool: dtype = check_dtype_on_file(path,header,columns)
            return load_ascii_grid(path,blocks,header,dtype,columns,swap,swap_directory)
        else:
            print 'ERROR ON "load_grid": blocks must be tuple of length 1,2 or 3.'
            return False
            
def swap_load_ascii_grib(path,counter,blocks,dtype='float32',swap_directory='TMP'):
    '''
    swap_load_ascii_grib(...)
        swap_load_ascii_grib(path,counter,blocks,dtype)
        
        Loads a grib ASCII (text) file into memory swap space. 
        
    Parameters
    ----------
    path : string
        String with file path.
        
    counter : int
        integer with the number of variables in file.
        
    blocks : tuple
        Tuple of integers with number of blocks in (X,Y,Z).
        
    dtype : string
        String with data type. Default is float32.
        
    swap_directory : string
        String with directory to build the swap file (if swap is True). 
        
    Returns
    -------
    out: numpy swap array
        4D numpy swap array with information on file (4th dimension has the size of
        the number of variables).
    
    See also
    --------
    load_grib, load_ascii_grib, swap_load_ascii_grib
    '''
    directory_list_of_files = os.listdir(swap_directory)
    chosen_name = 'grib_'+str(np.random.randint(1000,9999))+'.dat'
    while chosen_name in directory_list_of_files:
        chosen_name = 'grib_'+str(np.random.randint(1000,9999))+'.dat'
    mem_grib = np.memmap(swap_directory+'\\'+chosen_name, dtype=dtype, mode='w+', shape=(blocks[0],blocks[1],blocks[2],counter))
    fid = open(path,'r')
    if dtype in ['float','float16','float32','float64']:
        for i in xrange(counter):
            fid.readline()
            for z in xrange(blocks[2]):
                for y in xrange(blocks[1]):
                    for x in xrange(blocks[0]):
                        mem_grib[x,y,z,counter] = np.float(fid.readline().replace('\n',''))
    elif dtype == 'bool':
        for i in xrange(counter):
            fid.readline()
            for z in xrange(blocks[2]):
                for y in xrange(blocks[1]):
                    for x in xrange(blocks[0]):
                        mem_grib[x,y,z,counter] = np.bool(np.int(fid.readline().replace('\n','')))
    elif dtype in ['int','int8','int16','int32','int64','uint8','uint16','uint32','uint64']:
        for i in xrange(counter):
            fid.readline()
            for z in xrange(blocks[2]):
                for y in xrange(blocks[1]):
                    for x in xrange(blocks[0]):
                        mem_grib[x,y,z,counter] = np.int(fid.readline().replace('\n',''))
    fid.close()
    return mem_grib
    
def load_ascii_grib(path,counter,blocks,dtype='float32'):
    '''
    load_ascii_grib(...)
        load_ascii_grib(path,counter,blocks,dtype)
        
        Loads a grib ASCII (text) file. 
        
    Parameters
    ----------
    path : string
        String with file path.
        
    counter : int
        integer with the number of variables in file.
        
    blocks : tuple
        Tuple of integers with number of blocks in (X,Y,Z).
        
    dtype : string
        String with data type. Default is float32.
        
    Returns
    -------
    out: numpy array
        4D numpy array with information on file (4th dimension has the size of
        the number of variables).
    
    See also
    --------
    load_grib, load_ascii_grib, swap_load_ascii_grib, load_npy_grid
    '''
    grib = np.zeros((blocks[0],blocks[1],blocks[2],counter),dtype=dtype)
    fid = open(path,'r')
    if dtype in ['float','float16','float32','float64']:
        for i in xrange(counter):
            fid.readline()
            for z in xrange(blocks[2]):
                for y in xrange(blocks[1]):
                    for x in xrange(blocks[0]):
                        grib[x,y,z,counter] = np.float(fid.readline().replace('\n',''))
    elif dtype == 'bool':
        for i in xrange(counter):
            fid.readline()
            for z in xrange(blocks[2]):
                for y in xrange(blocks[1]):
                    for x in xrange(blocks[0]):
                        grib[x,y,z,counter] = np.bool(np.int(fid.readline().replace('\n','')))
    elif dtype in ['int','int8','int16','int32','int64','uint8','uint16','uint32','uint64']:
        for i in xrange(counter):
            fid.readline()
            for z in xrange(blocks[2]):
                for y in xrange(blocks[1]):
                    for x in xrange(blocks[0]):
                        grib[x,y,z,counter] = np.int(fid.readline().replace('\n',''))
    fid.close()
    return grib
    
            
def load_grib(path,dtype='float32',swap=False,swap_directory='TMP'):
    '''
    load_grib(...)
        load_grib(path,dtype,swap,swap_directory)
        
        Loads a grib ASCII (text) file. If file path ends in .npy its assumed
        its a numpy binary (NPY file). GRIB is special format file a bit
        different from GEOEAS format (common in geostatistics).
        
    Parameters
    ----------
    path : string
        String with file path.
        
    dtype : string
        String with data type. Default is float32.
        
    swap : bool
        Boolean value indicating if output variable should be memory swap.
        Default is False.
        
    swap_directory : string
        String with directory to build the swap file (if swap is True).    
        
    Returns
    -------
    out: numpy array (or swap)
        4D numpy array with information on file (4th dimension has the size of
        the number of variables).
    
    See also
    --------
    load_grib, load_ascii_grib, swap_load_ascii_grib
    '''
    if path[-4:] == '.npy':
        return load_npy_grid(path,swap,swap_directory)
    else:
        fid = open(path,'r')
        appex0 = fid.readline()
        appex1 = np.int_(appex0.replace('\n','').split())
        if len(appex1)==2:
            blocks = (appex1[0],appex1[1],1)
        elif len(appex1)==3:
            blocks = (appex1[0],appex1[1],appex1[2])
        else:
            print 'ERROR ON "load_grib": dimension of blocks not recognized in first line. Either 2 dimensions or 3 dimensions.'
            return False
        fid.close()
        counter = 0
        with open(path, 'r') as f:
            for line in f:
                if line == appex0: counter = counter + 1
        fid = open(path,'r')
        if swap:
            __manage_directory__(swap_directory)
            return swap_load_ascii_grib(path,counter,blocks,dtype,swap_directory='TMP')
        else:
            return load_ascii_grib(path,counter,blocks,dtype)
            
            
def load_npy_point(path,swap=False,swap_directory='TMP'):
    '''
    load_npy_point(...)
        load_npy_point(path,swap,swap_directory)
        
        Loads a point NPY (numpy binary) file. The coordinate columns for X,Y
        and Z are assumed to be correctly in the columns (Y) of the NPY
        array (0,1,2).
        
    Parameters
    ----------
    path : string
        String with file path.
        
    swap : bool
        Boolean value indicating if output variable should be memory swap.
        Default is False.
        
    swap_directory : string
        String with directory to build the swap file (if swap is True).      
        
    Returns
    -------
    out: numpy array (or swap)
        2D numpy array with information on file.
    
    See also
    --------
    load_ascii_point, load_npy_point, load_point
    '''
    if swap:
        directory_list_of_files = os.listdir(swap_directory)
        chosen_name = 'point_'+str(np.random.randint(1000,9999))+'.dat'
        while chosen_name in directory_list_of_files:
            chosen_name = 'point_'+str(np.random.randint(1000,9999))+'.dat'
        point = np.load(path)
        mem_point = np.memmap(swap_directory+'\\'+chosen_name, dtype=point.dtype, mode='w+', shape=point.shape, order='F')
        mem_point[:] = point[:]
        del(point)
        return mem_point
    else:
        return np.load(path)
    
def load_ascii_point(path,coordinate_columns = (1,2,3),header=0,dtype='float32',swap=False,swap_directory='TMP'):
    '''
    load_ascii_point(...)
        load_ascii_point(path,coordinate_columns,header,dtype,swap,swap_directory)
        
        Loads a point ASCII (text) file. 
        
    Parameters
    ----------
    path : string
        String with file path.
        
    coordinate_columns : tuple
        Tuple of integers giving the columns (starting in 1) of the X,Y and Z
        information. Default is (1,2,3) for (X,Y,Z).
        
    header : int
        Integer with the size of header rows.
        
    dtype : string
        String with data type. Default is float32.
        
    swap : bool
        Boolean value indicating if output variable should be memory swap.
        Default is False.
        
    swap_directory : string
        String with directory to build the swap file (if swap is True).      
        
    Returns
    -------
    out: numpy array (or swap)
        2D numpy array with information on file (with X,Y,Z columns in 0,1,2).
    
    See also
    --------
    load_ascii_point, load_npy_point, load_point
    '''
    fid = open(path,'r')
    for i in xrange(header): fid.readline()
    point = np.loadtxt(fid,dtype=dtype)
    coordinate_columns = [coordinate_columns[0],coordinate_columns[1],coordinate_columns[2]]
    #lnumber = 0
    #for i in xrange(len(coordinate_columns)):
    #    if coordinate_columns[i]!=0: lnumber = lnumber+1
    number = coordinate_columns.count(0)
    appex_point = np.zeros((point.shape[0],point.shape[1]+number),dtype=dtype)
    counter = 0
    for i in xrange(len(coordinate_columns)):
        if coordinate_columns[i] != 0 and coordinate_columns[i]-1 < point.shape[1]:
            appex_point[:,counter] = point[:,coordinate_columns[i]-1]
        counter = counter + 1
    counter = 3
    for i in xrange(point.shape[1]):
        if i+1 not in coordinate_columns:
            #print counter,i,point.shape,appex_point.shape
            appex_point[:,counter] = point[:,i]
            counter = counter + 1           
    fid.close()
    if swap:
        directory_list_of_files = os.listdir(swap_directory)
        chosen_name = 'point_'+str(np.random.randint(1000,9999))+'.dat'
        while chosen_name in directory_list_of_files:
            chosen_name = 'point_'+str(np.random.randint(1000,9999))+'.dat'
        mem_point = np.memmap(swap_directory+'\\'+chosen_name, dtype=point.dtype, mode='w+', shape=point.shape, order='F')
        mem_point[:] = appex_point[:]
        return mem_point
    else:
        return appex_point
            
def load_point(path,coordinate_columns=(1,2,3),dtype='float32',swap=False,swap_directory='TMP',at_least=3):
    '''
    load_point(...)
        load_point(path,coordinate_columns,dtype,swap,swap_directory,at_least)
        
        Loads a point ASCII (text) file. If file path ends in .npy its assumed
        its a numpy binary (NPY file). Tests to number and columns (whitespace
        separator) are made.
        
    Parameters
    ----------
    path : string
        String with file path.
        
    coordinate_columns : tuple
        Tuple of integers giving the columns (starting in 1) of the X,Y and Z
        information. Default is (1,2,3) for (X,Y,Z).
        
    dtype : string
        String with data type. Default is float32.
        
    swap : bool
        Boolean value indicating if output variable should be memory swap.
        Default is False.
        
    swap_directory : string
        String with directory to build the swap file (if swap is True).    
        
    at_least : int
        Integer giving the number of lines in row which are digit. If it gets
        to this number than its assumed the information is no longer header
        information. Default is 3.
        
    Returns
    -------
    out: numpy array (or swap)
        2D numpy array with information on file (with X,Y,Z columns in 0,1,2).
    
    See also
    --------
    load_ascii_point, load_npy_point, load_point
    '''
    if path[-4:] == '.npy':
        return load_npy_point(path,swap,swap_directory)
    else:
        header = check_header_on_file(path,at_least)
        columns = check_number_of_columns(path,header)
        if type(dtype)==bool: dtype = check_dtype_on_file(path,header,columns)
        return load_ascii_point(path,coordinate_columns,header,dtype,swap,swap_directory)
        
def load_ascii_data(path,header=0,dtype='float32',swap=False,swap_directory='TMP'):
    '''
    load_ascii_data(...)
        load_ascii_data(path,header,dtype,swap,swap_directory)
        
        Loads a data (non-spatial) ASCII (text) file. 
        
    Parameters
    ----------
    path : string
        String with file path.
        
    header : int
        Integer with the size of header rows.    
    
    dtype : string
        String with data type. Default is float32.
        
    swap : bool
        Boolean value indicating if output variable should be memory swap.
        Default is False.
        
    swap_directory : string
        String with directory to build the swap file (if swap is True).    
        
    Returns
    -------
    out: numpy array
        2D numpy array with information on file.
    
    See also
    --------
    load_ascii_data, load_npy_point, load_data
    '''
    fid = open(path,'r')
    for i in xrange(header): fid.readline()
    data = np.loadtxt(fid,dtype=dtype)
    fid.close()
    if swap:
        directory_list_of_files = os.listdir(swap_directory)
        chosen_name = 'data_'+str(np.random.randint(1000,9999))+'.dat'
        while chosen_name in directory_list_of_files:
            chosen_name = 'data_'+str(np.random.randint(1000,9999))+'.dat'
        mem_data = np.memmap(swap_directory+'\\'+chosen_name, dtype=data.dtype, mode='w+', shape=data.shape, order='F')
        mem_data[:] = data[:]
        #del(point)
        return mem_data
    else:
        return data
        
def load_data(path,dtype='float32',swap=False,swap_directory='TMP',at_least=3):
    '''
    load_data(...)
        load_data(path,dtype,swap,swap_directory,at_least)
        
        Loads a data (non-spatial) file. If file ends with .npy its assumed
        its a numpy binary (NPY) file and load_npy_point is used.
        
    Parameters
    ----------
    path : string
        String with file path.  
    
    dtype : string
        String with data type. Default is float32.
        
    swap : bool
        Boolean value indicating if output variable should be memory swap.
        Default is False.
        
    swap_directory : string
        String with directory to build the swap file (if swap is True).
        
    at_least : int
        Integer giving the number of lines in row which are digit. If it gets
        to this number than its assumed the information is no longer header
        information. Default is 3.
        
    Returns
    -------
    out: numpy array
        2D numpy array with information on file (assumed no spatial information).
    
    See also
    --------
    load_ascii_data, load_npy_point, load_data
    '''
    if path[-4:] == '.npy':
        load_npy_point(path,swap,swap_directory)
    else:
        header = check_header_on_file(path,at_least)
        columns = check_number_of_columns(path,header)
        if type(dtype)==bool: dtype = check_dtype_on_file(path,header,columns)
        return load_ascii_data(path,header,dtype,swap,swap_directory)
        
def load_flexible_data(path,swap=False,swap_directory='TMP',at_least=3):
    '''
    load_flexible_data(...)
        load_flexible_data(path,swap,swap_directory,at_least)
        
        Loads a data (non-spatial) file. The file can have string columns.
        Memory swapping not implemented.
        
    Parameters
    ----------
    path : string
        String with file path.  
        
    swap : bool
        Boolean value indicating if output variable should be memory swap.
        Default is False.
        
    swap_directory : string
        String with directory to build the swap file (if swap is True).
        
    at_least : int
        Integer giving the number of lines in row which are digit. If it gets
        to this number than its assumed the information is no longer header
        information. Default is 3.
        
    Returns
    -------
    out: numpy array
        2D numpy array with information on file (assumed no spatial information).
    
    See also
    --------
    load_ascii_data, load_npy_point, load_data
    '''
    header = check_header_on_flexible_file(path,at_least)
    columns = check_number_of_columns(path,header)
    fid = open(path,'r')
    for i in xrange(header): fid.readline()
    data = np.loadtxt(fid,dtype='|S32')
    fid.close()
    return data
        
def load_indicator_block_file(path,bins,dtype='float32'):
    '''
    load_indicator_block_file(...)
        load_indicator_block_file(path,bins,dtype)
        
        Loads a block ASCII (text) file whose variables are indicator.
        
    Parameters
    ----------
    path : string
        String with file path.
        
    bins : int
        Number of indicator variables in file.
    
    dtype : string
        String with data type. Default is float32.
        
    Returns
    -------
    out: tuple
        Tuple with arrays for block-data (spatial), value(s), and error.
        (blocks,value,error)
    
    See also
    --------
    load_indicator_block_file,load_continuous_block_file,load_block
    '''
    fid = open(path,'r')
    fid.readline()
    number = int(fid.readline())
    number_loose = np.zeros(number,dtype='int')
    fid.readline()
    for i in xrange(number-1):
        for j in xrange(2): fid.readline()
        counter = 0
        while fid.readline().split()[0].isdigit():
            counter = counter + 1
        number_loose[i] = counter
    counter = 0
    for j in xrange(2): fid.readline()
    while fid.readline().split()!=[]:
        counter = counter + 1
    number_loose[number-1] = counter
    fid.close()
    fid = open(path,'r')
    for i in xrange(5): fid.readline()
    cols = len(fid.readline().split())
    fid.close()
    blocks = np.zeros(number,dtype='object')
    error = np.zeros(number,dtype=dtype)
    value = np.zeros((number,bins),dtype=dtype)
    for i in xrange(number):
        blocks[i] = np.zeros((number_loose[i],cols),dtype=dtype)
    fid = open(path,'r')
    for j in xrange(3): fid.readline()
    for i in xrange(number):
        value[i,:] = np.float_(fid.readline().split(';'))
        error[i] = np.float(fid.readline())
        for k in xrange(number_loose[i]):
            blocks[i][k,:] = np.int_(fid.readline().split())
        fid.readline()
    fid.close()
    return (blocks,value,error)
    
def load_continuous_block_file(path,dtype):
    '''
    load_continuous_block_file(...)
        load_continuous_block_file(path,dtype)
        
        Loads a block ASCII (text) file with continuous variable.
        
    Parameters
    ----------
    path : string
        String with file path.
    
    dtype : string
        String with data type. Default is float32.
        
    Returns
    -------
    out: tuple
        Tuple with arrays for block-data (spatial), value, and error.
        (blocks,value,error)
    
    See also
    --------
    load_indicator_block_file,load_continuous_block_file,load_block
    '''
    fid = open(path,'r')
    fid.readline()
    number = int(fid.readline())
    number_loose = np.zeros(number,dtype='int')
    fid.readline()
    for i in xrange(number-1):
        for j in xrange(2): fid.readline()
        counter = 0
        while fid.readline().split()[0].replace(".", "", 1).isdigit():
            counter = counter + 1
        number_loose[i] = counter
    counter = 0
    for j in xrange(2): fid.readline()
    while fid.readline().split()!=[]:
        counter = counter + 1
    number_loose[number-1] = counter
    fid.close()
    fid = open(path,'r')
    for i in xrange(5): fid.readline()
    cols = len(fid.readline().split())
    fid.close()
    blocks = np.zeros(number,dtype='object')
    error = np.zeros(number,dtype=dtype)
    value = np.zeros(number,dtype=dtype)
    for i in xrange(number):
        blocks[i] = np.zeros((number_loose[i],cols),dtype=dtype)
    fid = open(path,'r')
    for j in xrange(3): fid.readline()
    for i in xrange(number):
        value[i] = np.float_(fid.readline())
        error[i] = np.float(fid.readline())
        for k in xrange(number_loose[i]):
            blocks[i][k,:] = np.int_(np.float_(fid.readline().split()))
        fid.readline()
    fid.close()
    return (blocks,value,error)
    
def __check_bins_on_file__(path):
    '''
    __check_bins_on_file__(...)
        __check_bins_on_file__(path)
        
        Checks how many bins exist on block file in path. It does not test
        if file is block file. That is assumed.
        
    Parameters
    ----------
    path : string
        String with file path.
        
    Returns
    -------
    out: int
        Integer with the number of bins in file.
    
    See also
    --------
    load_indicator_block_file,load_continuous_block_file,__check_bins_on_file__,
    __check_if_indicator__,load_block
    '''
    fid = open(path,'r')
    for i in xrange(3): fid.readline()
    bins = len(fid.readline().split(';'))
    fid.close()
    return bins
    
def __check_if_indicator__(path):
    '''
    __check_if_indicator__(...)
        __check_if_indicator__(path)
        
        Checks if block file on path is indicator. It does not test
        if file is block file. That is assumed.
        
    Parameters
    ----------
    path : string
        String with file path.
        
    Returns
    -------
    out: bool
        Boolean value True or False to wether the block file is indicator or not.
    
    See also
    --------
    load_indicator_block_file,load_continuous_block_file,__check_bins_on_file__,
    __check_if_indicator__,load_block
    '''
    fid = open(path,'r')
    for i in xrange(3): fid.readline()
    bins = len(fid.readline().split(';'))
    if bins==1:
        return False
    else:
        return True
        
def load_block(path,dtype='float32',swap=False,swap_directory='TMP'):
    '''
    load_block(...)
        load_block(path,dtype,swap,swap_directory)
        
        Loads a block file. NPY not working for blocks. Block files have a
        special format with an expected header.
        
    Parameters
    ----------
    path : string
        String with file path.  
    
    dtype : string
        String with data type. Default is float32.
        
    swap : bool
        Boolean value indicating if output variable should be memory swap.
        Default is False.
        
    swap_directory : string
        String with directory to build the swap file (if swap is True).
        
    Returns
    -------
    out: tuple
        Tuple with arrays for block-data (spatial), value, and error.
        (blocks,value,error)
    
    See also
    --------
    load_indicator_block_file,load_continuous_block_file,__check_bins_on_file__,
    __check_if_indicator__,load_block
    '''
    if path[-4:] == '.npy':
        print 'ERROR ON "load_block": numpy binary support for blocks does not exist in this version.'
        return False
    if swap:
        print 'ERROR ON "load_block": memory swapping support for blocks does not exist in this version.'
    indicator = __check_if_indicator__(path)
    if indicator:
        bins = __check_bins_on_file__(path)
        return load_indicator_block_file(path,bins,dtype)
    else:
        return load_continuous_block_file(path,dtype)
    
def save_grid(grid,opath='default_grid_name.prn',fmt='%10.3f',header=False):
    '''
    save_grid(...)
        save_grid(grid,opath,fmt,header)
        
        Saves a grid (mesh) to file in opath. If opath ends in .npy that it is
        assumed the user wants a numpy binary file.
        
    Parameters
    ----------
    
    grid : numpy array
        Numpy array to be saved to file.    
    
    opath : string
        String with file (o)utput path. Default is default_grid_name.prn.
    
    fmt : string
        String data format. Default is 10 characters with three digits beyhond
        the decimal separator.
        
    header : string
        string with header to be written. If several rows exist than they must
        be on the same string. Default is boolean False meaning no header.
        
    Returns
    -------
    out: boolen
        Boolean value indicating if operation worked or not (only for True).
    
    See also
    --------
    save_grid,save_grib,save_point,save_data,save_continuous_block,
    save_indicator_block,save_grid_by_dictionary
    '''
    if opath[-4:] == '.npy':
        np.save(opath,grid)
    else:
        fid = open(opath,'w')
        if type(header)!=bool: 
            if header[-1:]=='\n': fid.write(header)
            else: fid.write(header+'\n')
            #print header
        if len(grid.shape)==3:
            for z in xrange(grid.shape[2]):
                for y in xrange(grid.shape[1]):
                    for x in xrange(grid.shape[0]):
                        fid.write((fmt+'\n')%grid[x,y,z])
            fid.close()
            return True
        elif len(grid.shape)==4:
            for z in xrange(grid.shape[2]):
                for y in xrange(grid.shape[1]):
                    for x in xrange(grid.shape[0]):
                        for d in xrange(grid.shape[3]):
                            fid.write((fmt+'          ')%grid[x,y,z,d])
                        fid.write('\n')
            fid.close()
            return True
        else:
            fid.close()
            print 'ERROR ON "save_grid": length of grid shape not recognized. It should tuple of size 3 or 4.'
            return False
            
def cerena_save_grid_by_dictionary(dictionary,opath='default_grid_name.prn',fmt='%10.3f',header=False):
    '''
    cerena_save_grid_by_dictionary(...)
        cerena_save_grid(dictionary,opath,fmt,header)
        
        Saves a grid (mesh) to file in opath. If opath ends in .npy that it is
        assumed the user wants a numpy binary file.
        
    Parameters
    ----------
    
    dictionary : numpy array
        Dictionary with several variables.    
    
    opath : string
        String with file (o)utput path. Default is default_grid_name.prn.
    
    fmt : string
        String data format. Default is 10 characters with three digits beyhond
        the decimal separator.
        
    header : string
        string with header to be written. If several rows exist than they must
        be on the same string. Default is boolean False meaning no header.
        
    Returns
    -------
    out: boolen
        Boolean value indicating if operation worked or not (only for True).
    
    See also
    --------
    save_grid,save_grib,save_point,save_data,save_continuous_block,
    save_indicator_block
    '''
    if opath[-4:] == '.npy':
        grid = []
        shape = dictionary[dictionary.keys()[0]].data.shape
        for v in dictionary.keys():
            grid.append(dictionary[v].data.reshape((shape[0],shape[1],shape[2],1)))
        grid = np.concatenate(grid,axis=3)
        np.save(opath,grid)
    else:
        fid = open(opath,'w')
        #print header,dictionary
        if type(header)!=bool: 
            if header[-1:]=='\n': fid.write(header)
            else: fid.write(header+'\n')
        shape = dictionary[dictionary.keys()[0]].data.shape
        for z in xrange(shape[2]):
            for y in xrange(shape[1]):
                for x in xrange(shape[0]):
                    for v in dictionary.keys():
                        fid.write((fmt+'          ')%dictionary[v].data.data[x,y,z])
                    fid.write('\n')
        fid.close()
        return True
            
def save_grib(grib,opath='default_grib_name.prn',fmt='%10.3f'):
    '''
    save_grib(...)
        save_grib(grib,opath,fmt,header)
        
        Saves a grib (GRIB format) to file in opath. If opath ends in .npy that it is
        assumed the user wants a numpy binary file.
        
    Parameters
    ----------
    
    grib : numpy array
        Numpy array to be saved to file.    
    
    opath : string
        String with file (o)utput path. Default is default_grib_name.prn.
    
    fmt : string
        String data format. Default is 10 characters with three digits beyhond
        the decimal separator.
        
    Returns
    -------
    out: boolen
        Boolean value indicating if operation worked or not (only for True).
    
    See also
    --------
    save_grid,save_grib,save_point,save_data,save_continuous_block,
    save_indicator_block
    '''
    if opath[-4:] == '.npy':
        np.save(opath,grib)
    else:
        fid = open(opath,'w')
        for i in xrange(grib.shape[3]):
            fid.write(str(grib.shape[0])+'     '+str(grib.shape[1])+'     '+str(grib.shape[2])+'\n')
            for z in xrange(grib.shape[2]):
                for y in xrange(grib.shape[1]):
                    for x in xrange(grib.shape[0]):
                        fid.write(fmt%grib[x,y,z,i]+'\n')
        fid.close()
    return True

def cerena_save_point_by_dictionary(dictionary,xc,yc,zc,opath='default_point_name.prn',fmt='%10.3f',header=False):
    '''
    cerena_save_point_by_dictionary(...)
        cerena_save_grid(dictionary,opath,fmt,header)
        
        Saves a point to file in opath. If opath ends in .npy that it is
        assumed the user wants a numpy binary file.
        
    Parameters
    ----------
    
    dictionary : numpy array
        Dictionary with several variables.    
    
    opath : string
        String with file (o)utput path. Default is default_grid_name.prn.
    
    fmt : string
        String data format. Default is 10 characters with three digits beyhond
        the decimal separator.
        
    header : string
        string with header to be written. If several rows exist than they must
        be on the same string. Default is boolean False meaning no header.
        
    Returns
    -------
    out: boolen
        Boolean value indicating if operation worked or not (only for True).
    
    See also
    --------
    save_grid,save_grib,save_point,save_data,save_continuous_block,
    save_indicator_block
    '''
    if opath[-4:] == '.npy':
        grid = []
        shape = dictionary[dictionary.keys()[0]].data.shape
        for v in dictionary.keys():
            grid.append(dictionary[v].data.reshape((shape[0],1)))
        grid = np.concatenate(grid,axis=1)
        np.save(opath,grid)
    else:
        fid = open(opath,'w')
        if type(header)!=bool: 
            if header[-1:]=='\n': fid.write(header)
            else: fid.write(header+'\n')
        shape = dictionary[dictionary.keys()[0]].data.shape
        for x in xrange(shape[0]):
            fid.write((fmt+'          '+fmt+'          '+fmt+'          ')%(xc[x],yc[x],zc[x]))
            for v in dictionary.keys():
                fid.write((fmt+'          ')%dictionary[v].data.data[x])
            fid.write('\n')
        fid.close()
        return True
        
def cerena_save_flexible_data_by_dictionary(dictionary,opath='default_data_name.prn',fmt='%10.3f',header=False):
    '''
    cerena_save_flexible_data_by_dictionary(...)
        cerena_save_flexible_data_by_dictionary(dictionary,opath,fmt,header)
        
        Saves a point to file in opath. If opath ends in .npy that it is
        assumed the user wants a numpy binary file.
        
    Parameters
    ----------
    
    dictionary : dictionary
        Dictionary with several variables.    
    
    opath : string
        String with file (o)utput path. Default is default_grid_name.prn.
    
    fmt : string
        String data format. Default is 10 characters with three digits beyhond
        the decimal separator.
        
    header : string
        string with header to be written. If several rows exist than they must
        be on the same string. Default is boolean False meaning no header.
        
    Returns
    -------
    out: boolen
        Boolean value indicating if operation worked or not (only for True).
    
    See also
    --------
    save_grid,save_grib,save_point,save_data,save_continuous_block,
    save_indicator_block
    '''
    if opath[-4:] == '.npy':
        grid = []
        shape = dictionary[dictionary.keys()[0]].data.shape
        for v in dictionary.keys():
            grid.append(dictionary[v].data.reshape((shape[0],1)))
        grid = np.concatenate(grid,axis=1)
        np.save(opath,grid)
    else:
        fid = open(opath,'w')
        if type(header)!=bool: 
            if header[-1:]=='\n': fid.write(header)
            else: fid.write(header+'\n')
        shape = dictionary[dictionary.keys()[0]].data.shape
        for x in xrange(shape[0]):
            #fid.write((fmt+'          '+fmt+'          '+fmt+'          ')%(xc[x],yc[x],zc[x]))
            for v in dictionary.keys():
                if dictionary[v].dtype!='string': fid.write((fmt+'          ')%dictionary[v].data[x])
                else: fid.write(dictionary[v].data.data[x]+'          ')
            fid.write('\n')
        fid.close()
        return True
                                
def save_point(point,opath='default_point_name.prn',fmt='%10.3f',header=False):
    '''
    save_point(...)
        save_point(point,opath,fmt,header)
        
        Saves a point data to file in opath. If opath ends in .npy that it is
        assumed the user wants a numpy binary file.
        
    Parameters
    ----------
    
    point : numpy array
        Numpy array to be saved to file.    
    
    opath : string
        String with file (o)utput path. Default is default_point_name.prn.
    
    fmt : string
        String data format. Default is 10 characters with three digits beyhond
        the decimal separator.
        
    header : string
        string with header to be written. If several rows exist than they must
        be on the same string. Default is boolean False meaning no header.
        
    Returns
    -------
    out: boolen
        Boolean value indicating if operation worked or not (only for True).
    
    See also
    --------
    save_grid,save_grib,save_point,save_data,save_continuous_block,
    save_indicator_block
    '''
    if opath[-4:] == '.npy':
        np.save(opath,point)
    else:
        fid = open(opath,'w')
        if type(header)!=bool: 
            if header[-1:]=='\n': fid.write(header)
            else: fid.write(header+'\n')
        np.savetxt(fid,point,fmt=fmt)
        fid.close()
        return True
        
def save_flexible_data(point,opath='default_data_name.prn',fmt='%10.3f',header=False):
    '''
    save_point(...)
        save_point(point,opath,fmt,header)
        
        Saves a point data to file in opath. If opath ends in .npy that it is
        assumed the user wants a numpy binary file.
        
    Parameters
    ----------
    
    point : numpy array
        Numpy array to be saved to file.    
    
    opath : string
        String with file (o)utput path. Default is default_point_name.prn.
    
    fmt : string
        String data format. Default is 10 characters with three digits beyhond
        the decimal separator.
        
    header : string
        string with header to be written. If several rows exist than they must
        be on the same string. Default is boolean False meaning no header.
        
    Returns
    -------
    out: boolen
        Boolean value indicating if operation worked or not (only for True).
    
    See also
    --------
    save_grid,save_grib,save_point,save_data,save_continuous_block,
    save_indicator_block
    '''
    if opath[-4:] == '.npy':
        np.save(opath,point)
    else:
        fid = open(opath,'w')
        if type(header)!=bool: 
            if header[-1:]=='\n': fid.write(header)
            else: fid.write(header+'\n')
        if type(point[0])!=str: np.savetxt(fid,point.data,fmt=fmt)
        else:
            for i in xrange(point.shape[0]):
                fid.write(point[i]+'\n')
        fid.close()
        return True
        
def save_data(data,opath='default_data_name.prn',fmt='%10.3f',header=False):
    '''
    save_data(...)
        save_data(data,opath,fmt,header)
        
        Saves data to file in opath. If opath ends in .npy that it is
        assumed the user wants a numpy binary file.
        
    Parameters
    ----------
    
    data : numpy array
        Numpy array to be saved to file.    
    
    opath : string
        String with file (o)utput path. Default is default_point_name.prn.
    
    fmt : string
        String data format. Default is 10 characters with three digits beyhond
        the decimal separator.
        
    header : string
        string with header to be written. If several rows exist than they must
        be on the same string. Default is boolean False meaning no header.
        
    Returns
    -------
    out: boolen
        Boolean value indicating if operation worked or not (only for True).
    
    See also
    --------
    save_grid,save_grib,save_point,save_data,save_continuous_block,
    save_indicator_block
    '''
    if opath[-4:] == '.npy':
        np.save(opath,data)
    else:
        fid = open(opath,'w')
        if type(header)==str: 
            if header[-1:]=='\n': fid.write(header)
            else: fid.write(header+'\n')
        np.savetxt(fid,data,fmt=fmt)
        fid.close()
        return True

def save_continuous_block(block,opath='default_continuous_block_name.prn',fmt='%10.3f'):
    '''
    save_continuous_block(...)
        save_continuous_block(block,opath,fmt)
        
        Saves continuous block (tuple) to file in opath.
        
    Parameters
    ----------
    
    block : tuple
        Tuple with (block point locations, value, error) information.    
    
    opath : string
        String with file (o)utput path. Default is default_point_name.prn.
    
    fmt : string
        String data format. Default is 10 characters with three digits beyhond
        the decimal separator.
        
    Returns
    -------
    out: boolen
        Boolean value indicating if operation worked or not (only for True).
    
    See also
    --------
    save_grid,save_grib,save_point,save_data,save_continuous_block,
    save_indicator_block
    '''
    fid = open(opath,'w')
    fid.write('BLOCK_FILE\n'+str(block[1].shape[0])+'\n')
    count = 0
    for k in xrange(block[1].shape[0]):
        fid.write('block #'+str(count)+'\n')
        fid.write(str(block[1][k])+'\n')
        fid.write(str(block[2][k])+'\n')
        for j in xrange(block[0][k].shape[0]): fid.write(str(block[0][k][j,0])+'     '+str(block[0][k][j,1])+'     '+str(block[0][k][j,2])+'\n')
        count = count + 1
    fid.close()
    return True
    
def save_indicator_block(block,opath='default_indicator_block_name.prn',fmt='%10.3f'):
    '''
    save_indicator_block(...)
        save_indicator_block(block,opath,fmt)
        
        Saves indicator block (tuple) to file in opath.
        
    Parameters
    ----------
    
    block : tuple
        Tuple with (block point locations, value, error) information.   
    
    opath : string
        String with file (o)utput path. Default is default_point_name.prn.
    
    fmt : string
        String data format. Default is 10 characters with three digits beyhond
        the decimal separator.
        
    Returns
    -------
    out: boolen
        Boolean value indicating if operation worked or not (only for True).
    
    See also
    --------
    save_grid,save_grib,save_point,save_data,save_continuous_block,
    save_indicator_block
    '''
    fid = open(opath,'w')
    fid.write('BLOCK_FILE\n'+str(block[1].shape[0])+'\n')
    count = 0
    for k in xrange(block[1].shape[0]):
        fid.write('block #'+str(count)+'\n')
        for j in xrange(block[1].shape[1]-1): fid.write('%i'%(block[1][k,j])+';')
        fid.write(str(block[1][k,-1])+'\n'+str(block[2][k])+'\n')
        for j in xrange(block[0][k].shape[0]): fid.write(str(block[0][k][j,0])+'     '+str(block[0][k][j,1])+'     '+str(block[0][k][j,2])+'\n')
        count = count + 1
    fid.close()
    return True

def save_block(block,opath='default_block_name.prn',fmt='%10.3f'):
    '''
    save_block(...)
        save_block(block,opath,fmt)
        
        Saves block (tuple) to file in opath. Tests if indicator or not.
        
    Parameters
    ----------
    
    block : tuple
        Tuple with (block point locations, value, error) information.    
    
    opath : string
        String with file (o)utput path. Default is default_point_name.prn.
    
    fmt : string
        String data format. Default is 10 characters with three digits beyhond
        the decimal separator.
        
    Returns
    -------
    out: boolen
        Boolean value indicating if operation worked or not (only for True).
    
    See also
    --------
    save_grid,save_grib,save_point,save_data,save_continuous_block,
    save_indicator_block
    '''
    if opath[-4:] == '.npy':
        print 'ERROR ON "save_block": numpy binary support for blocks does not exist in this version.'
        return False
    else:
        if len(block[1].shape)==1:
            return save_continuous_block(block,opath,fmt)
        elif len(block[1].shape)==2:
            return save_indicator_block(block,opath,fmt)
        else:
            print 'ERROR ON "save_block": values (block[1]) shape length not recognized. If 1 it a continuous variable, if 2 indicator.'
            return False
            
def determine_header_format(path,at_least):
    '''
    determine_header_format(...)
        determine_header_format(path,at_least)
        
        Determines if ASCII file has type of header that could be unknown, cmrp
        or geoeas.
        
    Parameters
    ----------
    
    path : string
        String with file path.    
    
    at_least : int
        Integer giving the number of lines in row which are digit. If it gets
        to this number than its assumed the information is no longer header
        information. Default is 3.
        
    Returns
    -------
    out: string
        String with type of header flag (unknown,cmrp,geoeas).
    
    See also
    --------
    determine_header_format,pygeo_determine_header_format,
    pygeo_get_names_from_header,pygeo_create_variables_names,
    pygeo_get_point_names_from_header,determine_special_char_separator,
    check_dtype_on_special_char_file      
    '''
    header  = check_header_on_file(path,at_least)
    columns = check_number_of_columns(path,header)
    flag = 'unknown'
    if columns+2==header:
        fid = open(path,'r')
        appex = fid.readline().replace('\n','').split('_')[1:]
        if len(appex) >= 3:
            cmrp_flag = True
            geoeas_flag = True
            for i in xrange(len(appex)):
                if not appex[i].replace('.','').isdigit():
                    cmrp_flag = False
                    break
            if cmrp_flag:
                number = len(appex)
                appex = fid.readline().replace('\n','')
                if not appex.isdigit():
                    cmrp_flag = False
                    geoeas_flag = False
                if geoeas_flag:
                    for i in xrange(columns):
                        if fid.readline().replace('\n','').replace('.','').isdigit():
                            cmrp_flag=False
                            geoeas_flag=False
            if cmrp_flag and number in [3,6,9]:                
                flag = 'cmrp'+str(number)
                return flag
            else:
                if geoeas_flag:
                    flag = 'geoeas'
                    return flag
                else:
                    return 'unknown'            
    else:
        return flag
        
def pygeo_determine_header_format(path,at_least):
    '''
    pygeo_determine_header_format(...)
        determine_header_format(path,at_least)
        
        Determines if ASCII file has type of header that could be unknown, cmrp
        or geoeas.
        
    Parameters
    ----------
    
    path : string
        String with file path.    
    
    at_least : int
        Integer giving the number of lines in row which are digit. If it gets
        to this number than its assumed the information is no longer header
        information. Default is 3.
        
    Returns
    -------
    out: string
        String with type of header flag (unknown,cmrp,geoeas).
    
    See also
    --------
    determine_header_format,pygeo_determine_header_format,
    pygeo_get_names_from_header,pygeo_create_variables_names,
    pygeo_get_point_names_from_header,determine_special_char_separator,
    check_dtype_on_special_char_file      
    '''
    header  = check_header_on_file(path,at_least)
    columns = check_number_of_columns(path,header)
    flag = 'unknown'
    if columns+2==header:
        fid = open(path,'r')
        if fid.readline().replace('\n','').replace('.','').replace('-','').isdigit(): flag = 'unknown'
        else: flag = 'geoeas'
        if fid.readline().replace('\n','').replace('.','').replace('-','').isdigit(): flag = 'geoeas'
        else: flag = 'unknown'
        for i in xrange(columns):
            if fid.readline().replace('\n','').replace('.','').replace('-','').isdigit(): flag = 'unknown'
            else: flag = 'geoeas'
    return flag
    
def pygeo_determine_header_in_flexible_format(path,at_least):
    '''
    pygeo_determine_header_in_flexible_format(...)
        determine_header_in_flexible_format(path,at_least)
        
        Determines if ASCII file has type of header that could be unknown, cmrp
        or geoeas.
        
    Parameters
    ----------
    
    path : string
        String with file path.    
    
    at_least : int
        Integer giving the number of lines in row which are digit. If it gets
        to this number than its assumed the information is no longer header
        information. Default is 3.
        
    Returns
    -------
    out: string
        String with type of header flag (unknown,cmrp,geoeas).
    
    See also
    --------
    determine_header_format,pygeo_determine_header_format,
    pygeo_get_names_from_header,pygeo_create_variables_names,
    pygeo_get_point_names_from_header,determine_special_char_separator,
    check_dtype_on_special_char_file      
    '''
    header  = check_header_on_flexible_file(path,at_least)
    columns = check_number_of_columns(path,header)
    flag = 'unknown'
    if columns+2==header:
        fid = open(path,'r')
        if fid.readline().replace('\n','').replace('.','').replace('-','').isdigit(): flag = 'unknown'
        else: flag = 'geoeas'
        if fid.readline().replace('\n','').replace('.','').replace('-','').isdigit(): flag = 'geoeas'
        else: flag = 'unknown'
        for i in xrange(columns):
            if fid.readline().replace('\n','').replace('.','').replace('-','').isdigit(): flag = 'unknown'
            else: flag = 'geoeas'
    return flag
            
        
def pygeo_get_names_from_header(path,columns):
    '''
    pygeo_get_names_from_header(...)
        pygeo_get_names_from_header(path,columns)
        
        Retrieves names name of data and names of variables from file.
        Assuming its a common geoeas (or similar) file format.
        
    Parameters
    ----------
    
    path : string
        String with file path.    
    
    columns : int
        Integer with the size of data columns.
        
    Returns
    -------
    out: list
        List with [name of data,list of variables names].
    
    See also
    --------
    determine_header_format,pygeo_determine_header_format,
    pygeo_get_names_from_header,pygeo_create_variables_names,
    pygeo_get_point_names_from_header,determine_special_char_separator,
    check_dtype_on_special_char_file      
    '''
    header_info = []
    fid = open(path,'r')
    header_info.append(fid.readline().replace('\n',''))
    fid.readline()
    variables = []
    for i in xrange(columns):
        variables.append(fid.readline().replace('\n',''))
    fid.close()
    header_info.append(variables)
    return header_info
    
def pygeo_create_variables_names(columns):
    '''
    pygeo_create_variables_names(...)
        pygeo_create_variables_names(columns)
        
        Creates a list of numbered variables names.
        
    Parameters
    ----------    
    
    columns : int
        Integer with the size of data columns.
        
    Returns
    -------
    out: list
        List with names fo variables.
    
    See also
    --------
    determine_header_format,pygeo_determine_header_format,
    pygeo_get_names_from_header,pygeo_create_variables_names,
    pygeo_get_point_names_from_header,determine_special_char_separator,
    check_dtype_on_special_char_file      
    '''
    variables = []
    for i in xrange(columns):
        variables.append('Variable_'+str(i))
    return variables
    
def pygeo_get_point_names_from_header(path,columns,coordinate_columns = (1,2,3)):
    '''
    pygeo_get_point_names_from_header(...)
        pygeo_get_point_names_from_header(path,columns)
        
        Retrieves names name of data and names of variables from file but
        avoinding the names of the coordinate columns. Only variables are
        retrieved. Assuming its a common geoeas (or similar) file format.
        
    Parameters
    ----------
    
    path : string
        String with file path.    
    
    columns : int
        Integer with the size of data columns.
        
    coordinate_columns : tuple
        Tuple of integers giving the columns (starting in 1) of the X,Y and Z
        information. Default is (1,2,3) for (X,Y,Z).
        
    Returns
    -------
    out: list
        List with [name of data,list of variables names].
    
    See also
    --------
    determine_header_format,pygeo_determine_header_format,
    pygeo_get_names_from_header,pygeo_create_variables_names,
    pygeo_get_point_names_from_header,determine_special_char_separator,
    check_dtype_on_special_char_file      
    '''
    header_info = []
    fid = open(path,'r')
    header_info.append(fid.readline().replace('\n',''))
    fid.readline()
    variables = []
    coordinate_columns = [coordinate_columns[0]-1,coordinate_columns[1]-1,coordinate_columns[2]-1]
    for i in xrange(0,columns):
        if i not in coordinate_columns: variables.append(fid.readline().replace('\n',''))
        else: fid.readline()
    fid.close()
    header_info.append(variables)
    return header_info
        
def determine_special_char_separator(path,at_least):
    '''
    determine_special_char_separator(...)
        determine_special_char_separator(path,at_least)
        
        Retrieves special char separator as well number of header rows and
        number of columns.
        
    Parameters
    ----------
    
    path : string
        String with file path.
        
    at_least : int
        Integer giving the number of lines in row which are digit. If it gets
        to this number than its assumed the information is no longer header
        information. Default is 3.
        
    Returns
    -------
    out: tuple
        tuple with (header number size, number of columns, char separator).
    
    See also
    --------
    determine_header_format,pygeo_determine_header_format,
    pygeo_get_names_from_header,pygeo_create_variables_names,
    pygeo_get_point_names_from_header,determine_special_char_separator,
    check_dtype_on_special_char_file      
    '''
    header  = check_header_special_char_on_file(path,at_least = 3)
    columns = check_number_of_columns_special_char(path,header[0],header[1])
    return (header[0],columns,header[1])
    
def check_dtype_on_special_char_file(path,header,columns,sep):
    '''
    check_dtype_on_special_char_file(...)
        check_dtype_on_special_char_file(path,header,columns,sep)
        
        Attempts recognition of data type on file. Either int or float32.
        
    Parameters
    ----------
    
    path : string
        String with file path.
        
    header : int
        Integer with the size of header rows.
        
    columns : int
        Integer with the size of data columns.
    
    sep : string
        String with character that acts like separator.
        
    Returns
    -------
    out: string
        String with data type. Either int or float32.
    
    See also
    --------
    determine_header_format,pygeo_determine_header_format,
    pygeo_get_names_from_header,pygeo_create_variables_names,
    pygeo_get_point_names_from_header,determine_special_char_separator,
    check_dtype_on_special_char_file      
    '''
    fid = open(path,'r')
    for i in xrange(header): fid.readline()
    if columns == 1:
        line = fid.readline().replace('\n','')
        if '.' in line:
            return 'float32'
        else:
            return 'int'
    else:
        line = fid.readline().replace('\n','').split(sep)
        for i in xrange(len(line)):
            if '.' in line[i]:
                return 'float32'
        return 'int'
        
def replace_char_in_file(ipath,opath,this=',',bythis='.'):
    '''
    replace_char_in_file(...)
        replace_char_in_file(ipath,opath,this,bythis)
        
        Replaces a character by another in a file.
        
    Parameters
    ----------
    
    ipath : string
        String with file (i)nput path.
        
    opath : string
        String with file (o)utput path. Default is default_grid_name.prn
        
    this : string
        String of character to be replaced. Default is ,
    
    bythis : string
        String of character to replace for. Default is .
        
    Returns
    -------
    out: boolen
        Boolean value indicating if operation worked or not (only for True).
    
    See also
    --------
    replace_char_in_file, convert_npy_to_ascii, convert_ascii_to_npy
    '''
    fid = open(opath,'w')
    with open(ipath, 'r') as f:
        for line in f:
            fid.write(line.replace(this,bythis))
    fid.close()
    return True
    
def convert_npy_to_ascii(ipath,opath,fmt='%10.3f'):
    '''
    convert_npy_to_ascii(...)
        convert_npy_to_ascii(ipath,opath,fmt)
        
        Converts input NPY file into output ASCII file.
        
    Parameters
    ----------
    
    ipath : string
        String with file (i)nput path.
        
    opath : string
        String with file (o)utput path. Default is default_grid_name.prn
        
    fmt : string
        String data format. Default is 10 characters with three digits beyhond
        the decimal separator.
        
    Returns
    -------
    out: boolen
        Boolean value indicating if operation worked or not (only for True).
    
    See also
    --------
    replace_char_in_file, convert_npy_to_ascii, convert_ascii_to_npy
    '''
    if ipath[-4:] != '.npy':
        print 'ERROR ON "convert_npy_to_ascii": ipath (input file) must be numpy binary with .npy file extension.'
        return False
    else:
        idata = np.load(ipath)
        if len(idata.shape) > 4:
            print 'ERROR ON "convert_npy_to_ascii": data on npy file cant have more than 4 dimensions.'
            return False
        elif len(idata.shape)==4:
            fid = open(opath,'w')
            fid.write('your_data_'+str(idata.shape[0])+'_'+str(idata.shape[1])+'_'+str(idata.shape[2])+'\n')
            fid.write('%i\n'%idata.shape[3])
            for i in xrange(idata.shape[3]):
                fid.write('variable'+str(i)+'\n')
            for z in xrange(idata.shape[2]):
                for y in xrange(idata.shape[1]):
                    for x in xrange(idata.shape[0]):
                        for i in xrange(idata.shape[3]):
                            fid.write(fmt%idata[x,y,z,i]+'     ')
                        fid.write('\n')
            fid.close()
            return True
        elif len(idata.shape)==3:
            fid = open(opath,'w')
            fid.write('your_data_'+str(idata.shape[0])+'_'+str(idata.shape[1])+'_'+str(idata.shape[2])+'\n1\nvariable0\n')
            for z in xrange(idata.shape[2]):
                for y in xrange(idata.shape[1]):
                    for x in xrange(idata.shape[0]):
                        fid.write(fmt%idata[x,y,z]+'     ')
                        fid.write('\n')
            fid.close()
            return True
        elif len(idata.shape)==2:
            fid = open(opath,'w')
            fid.write('your_data\n2\nvariable0\nvariable1\n')
            np.savetxt(fid,idata,fmt=fmt)
            fid.close()
            return True
        elif len(idata.shape)==1:
            fid = open(opath,'w')
            fid.write('your_data\n2\nvariable0\nvariable1\n')
            np.savetxt(fid,idata,fmt=fmt)
            fid.close()
            return True
            
def convert_ascii_to_npy(ipath,opath,data_type='grid',dtype='float32',blocks=(1,1,1),at_least=3):
    '''
    convert_ascii_to_npy(...)
        convert_ascii_to_npy(ipath,opath,data_type,dtype,blocks,at_least)
        
        Converts input NPY file into output ASCII file.
        
    Parameters
    ----------
    
    ipath : string
        String with file (i)nput path.
        
    opath : string
        String with file (o)utput path. Default is default_grid_name.prn
        
    data_type : string
        String of the type of data to be converted (grid,point,data). Default
        is grid.
        
    dtype : string
        String with data type. Default is float32.
        
    blocks : tuple
        Tuple with number of blocks in (X,Y,Z). Default is (1,1,1).
        
    at_least : int
        Integer giving the number of lines in row which are digit. If it gets
        to this number than its assumed the information is no longer header
        information. Default is 3.
        
    Returns
    -------
    out: boolen
        Boolean value indicating if operation worked or not (only for True).
    
    See also
    --------
    replace_char_in_file, convert_npy_to_ascii, convert_ascii_to_npy
    '''
    if opath[-4:] != '.npy':
        print 'ERROR ON "convert_ascii_to_npy": opath (output file path) must have numpy binary file extension (.npy).'
        return False
    else:
        header = check_header_on_file(ipath,at_least)
        columns = check_number_of_columns(ipath,header)
        if type(dtype)==bool: dtype = check_dtype_on_file(ipath,header,columns)
        if data_type == 'grid':
            if columns > 1:
                grid = np.zeros((blocks[0],blocks[1],blocks[2],columns),dtype=dtype)
                if dtype in ['float','float16','float32','float64']:
                    fid = open(ipath,'r')
                    for i in xrange(header): fid.readline()
                    for z in xrange(blocks[2]):
                        for y in xrange(blocks[1]):
                            for x in xrange(blocks[0]):
                                appex = fid.readline().replace('\n','').split()
                                grid[x,y,z,:] = np.float_(appex[:])
                        fid.close()
                    np.save(opath,grid)
                elif dtype == 'bool':
                    fid = open(ipath,'r')
                    for i in xrange(header): fid.readline()
                    if columns > 1:
                        grid = np.zeros((blocks[0],blocks[1],blocks[2],columns),dtype=dtype)
                        for z in xrange(blocks[2]):
                            for y in xrange(blocks[1]):
                                for x in xrange(blocks[0]):
                                    appex = fid.readline().replace('\n','').split()
                                    grid[x,y,z,:] = np.bool_(appex[:])
                    fid.close()
                    np.save(opath,grid)
                elif dtype in ['int','int8','int16','int32','int64','uint8','uint16','uint32','uint64']:
                    fid = open(ipath,'r')
                    for i in xrange(header): fid.readline()
                    if columns > 1:
                        grid = np.zeros((blocks[0],blocks[1],blocks[2],columns),dtype=dtype)
                        for z in xrange(blocks[2]):
                            for y in xrange(blocks[1]):
                                for x in xrange(blocks[0]):
                                    appex = fid.readline().replace('\n','').split()
                                    grid[x,y,z,:] = np.int_(appex[:])
                    fid.close()
                    np.save(opath,grid)
            else:
                grid = np.zeros((blocks[0],blocks[1],blocks[2]),dtype=dtype)
                if dtype in ['float','float16','float32','float64']:
                    fid = open(ipath,'r')
                    for i in xrange(header): fid.readline()
                    for z in xrange(blocks[2]):
                        for y in xrange(blocks[1]):
                            for x in xrange(blocks[0]):
                                appex = fid.readline().replace('\n','')
                                grid[x,y,z] = np.float(appex)
                        fid.close()
                    np.save(opath,grid)
                elif dtype == 'bool':
                    fid = open(ipath,'r')
                    for i in xrange(header): fid.readline()
                    if columns > 1:
                        grid = np.zeros((blocks[0],blocks[1],blocks[2]),dtype=dtype)
                        for z in xrange(blocks[2]):
                            for y in xrange(blocks[1]):
                                for x in xrange(blocks[0]):
                                    appex = fid.readline().replace('\n','')
                                    grid[x,y,z] = np.bool_(appex)
                    fid.close()
                    np.save(opath,grid)
                elif dtype in ['int','int8','int16','int32','int64','uint8','uint16','uint32','uint64']:
                    fid = open(ipath,'r')
                    for i in xrange(header): fid.readline()
                    if columns > 1:
                        grid = np.zeros((blocks[0],blocks[1],blocks[2],columns),dtype=dtype)
                        for z in xrange(blocks[2]):
                            for y in xrange(blocks[1]):
                                for x in xrange(blocks[0]):
                                    appex = fid.readline().replace('\n','')
                                    grid[x,y,z] = np.int_(appex)
                    fid.close()
                    np.save(opath,grid)
        elif data_type=='point':
            fid = open(ipath,'r')
            for i in xrange(header): fid.readline()
            point = np.loadtxt(fid,dtype=dtype)
            np.save(opath,point)
        elif data_type=='data':
            fid = open(ipath,'r')
            for i in xrange(header): fid.readline()
            point = np.loadtxt(fid,dtype=dtype)
            np.save(opath,point)
        elif data_type=='grib':
            grib = load_grib(ipath,dtype='float32',swap=False,swap_directory='TMP',at_least=3)
            np.save(opath,grib)
    return True
        
def swap_load_ascii_multiple_special_char_grid(path,blocks,header=0,dtype='float32',columns = 2,sep=';',swap_directory='TMP'):
    '''
    swap_load_ascii_multiple_special_char_grid(...)
        swap_load_ascii_multiple_special_char_grid(path,blocks,header,dtype,columns,sep,swap_directory)
        
        Loads a multiple (multiple columns) grid (mesh) ASCII file into a memory
        swap space. The separator for data in file is an argument.
        
    Parameters
    ----------
    path : string
        String with file path.
        
    blocks : tuple
        Tuple with number of blocks in (X,Y,Z).
        
    header : int
        Integer with the size of header rows.
        
    dtype : string
        Memory swap space data type. Default is float32.
        
    columns : int
        Integer with the size of data columns. Default is 2.
        
    sep : string
        String with character that acts like separator. Default is ;
        
    swap_directory : string
        String with directory to build the swap file.
        
    Returns
    -------
    out: swap array
        4D numpy swap array with information on file (4th dimension is given by
        number of variables).
    
    See also
    --------
    swap_load_ascii_single_grid,load_ascii_single_grid,load_ascii_multiple_grid
    swap_load_npy_grid,load_ascii_grid,load_npy_grid,load_grid
    '''
    fid = open(path,'r')
    for i in xrange(header): fid.readline()
    directory_list_of_files = os.listdir(swap_directory)
    chosen_name = 'grid_'+str(np.random.randint(1000,9999))+'.dat'
    while chosen_name in directory_list_of_files:
        chosen_name = str(np.random.randint(1000,9999))+'.dat'
    mem_grid = np.memmap(swap_directory+'\\'+chosen_name, dtype=dtype, mode='w+', shape=(blocks[0],blocks[1],blocks[2],columns), order='F')
    if dtype in ['float','float16','float32','float64']:
        for z in xrange(blocks[2]):
            for y in xrange(blocks[1]):
                for x in xrange(blocks[0]):
                    mem_grid[x,y,z,:] = np.float_(fid.readline().split(sep))
    elif dtype == 'bool':
        for z in xrange(blocks[2]):
            for y in xrange(blocks[1]):
                for x in xrange(blocks[0]):
                    mem_grid[x,y,z,:] = np.bool_(np.int_(fid.readline().split(sep)))
    elif dtype in ['int','int8','int16','int32','int64','uint8','uint16','uint32','uint64']:
        for z in xrange(blocks[2]):
            for y in xrange(blocks[1]):
                for x in xrange(blocks[0]):
                    mem_grid[x,y,z] = np.int_(fid.readline().split(sep))
    fid.close()
    return mem_grid
        
def load_ascii_multiple_special_char_grid(path,blocks,header=0,dtype='float32',columns=2,sep=';'):
    '''
    load_ascii_nultiple_special_char_grid(...)
        load_ascii_nultiple_special_char_grid(path,blocks,header,dtype,columns,sep)
        
        Loads a multiple (multiple columns) grid (mesh) ASCII file. The
        separator for data in file is an argument.
        
    Parameters
    ----------
    path : string
        String with file path.
        
    blocks : tuple
        Tuple with number of blocks in (X,Y,Z).
        
    header : int
        Integer with the size of header rows.
        
    dtype : string
        String with data type. Default is float32.
        
    columns : int
        Integer with the size of data columns. Default is 2.
        
    sep : string
        String with character that acts like separator. Default is ;
        
    Returns
    -------
    out: numpy array
        4D numpy array with information on file (4th dimension is given by
        number of variables).
    
    See also
    --------
    swap_load_ascii_single_grid,swap_load_ascii_nultiple_grid,load_ascii_single_grid
    swap_load_npy_grid,load_ascii_grid,load_npy_grid,load_grid
    '''
    fid = open(path,'r')
    for i in xrange(header): fid.readline()
    grid = np.zeros((blocks[0],blocks[1],blocks[2],columns),dtype=dtype,order='F')
    if dtype in ['float','float16','float32','float64']:
        for z in xrange(blocks[2]):
            for y in xrange(blocks[1]):
                for x in xrange(blocks[0]):
                    grid[x,y,z,:] = np.float_(fid.readline().split(sep))
    elif dtype == 'bool':
        for z in xrange(blocks[2]):
            for y in xrange(blocks[1]):
                for x in xrange(blocks[0]):
                    grid[x,y,z,:] = np.bool_(np.int_(fid.readline().split(sep)))
    elif dtype in ['int','int8','int16','int32','int64','uint8','uint16','uint32','uint64']:
        for z in xrange(blocks[2]):
            for y in xrange(blocks[1]):
                for x in xrange(blocks[0]):
                    grid[x,y,z,:] = np.int_(fid.readline().split(sep))
    fid.close()
    return grid
        
def load_ascii_special_char_grid(path,blocks,header=0,dtype='float32',columns=1,sep=' ',swap=False,swap_directory='TMP'):
    '''
    load_ascii_special_char_grid(...)
        load_ascii_special_char_grid(path,blocks,header,dtype,columns,sep,swap,swap_directory)
        
        Load an ASCII grid (mesh) from a file. The separator for data in file
        is an argument.
        
    Parameters
    ----------
    path : string
        String with file path.
        
    blocks : tuple
        Tuple with number of blocks in (X,Y,Z).
        
    header : int
        Integer with the size of header rows.
        
    dtype : string
        String with data type. Default is float32.
        
    columns : int
        Integer with the size of data columns. Default is 1 (single column).
        
    sep : string
        String with character that acts like separator. Default is ;
        
    swap : bool
        Boolean value indicating if output variable should be memory swap.
        Default is False.
        
    swap_directory : string
        String with directory to build the swap file (if swap is True).
        
    Returns
    -------
    out: numpy swap array (or swap)
        3D/4D numpy array with information on file (4th dimension is given by
        number of variables).
    
    See also
    --------
    swap_load_ascii_single_grid,swap_load_ascii_nultiple_grid,load_ascii_single_grid
    load_ascii_multiple_grid,swap_load_npy_grid,load_npy_grid,load_grid
    '''
    if swap:
        __manage_directory__(swap_directory)
        if columns == 1:
            return swap_load_ascii_single_grid(path,blocks,header,dtype,swap_directory)
        else:
            return swap_load_ascii_multiple_special_char_grid(path,blocks,header,dtype,columns,sep,swap_directory)
    else:
        if columns == 1:
            return load_ascii_single_grid(path,blocks,header,dtype)
        else:
            return load_ascii_multiple_special_char_grid(path,blocks,header,dtype,columns,sep)
    
def load_special_char_grid(path,blocks = (1,1,1),dtype='float32',swap=False,swap_directory='TMP',at_least=3):
    '''
    load_special_char_grid(...)
        load_special_char_grid(path,blocks,header,dtype,columns,swap,swap_directory)
        
        Loads a grid (mesh) from a file. If file path ends in .npy its assumed
        its a numpy binary (NPY file). The separator for data in file
        is accepted if i whitespace or ; or , or _ .
        
    Parameters
    ----------
    path : string
        String with file path.
        
    blocks : tuple
        Tuple with number of blocks in (X,Y,Z).
        
    dtype : string
        String with data type. Default is float32.
        
    swap : bool
        Boolean value indicating if output variable should be memory swap.
        Default is False.
        
    swap_directory : string
        String with directory to build the swap file (if swap is True).
        Default is TMP.
        
    at_least : int
        Integer giving the number of lines in row which are digit. If it gets
        to this number than its assumed the information is no longer header
        information. Default is 3.
        
    Returns
    -------
    out: numpy swap array (or swap)
        3D/4D numpy array with information on file (4th dimension is given by
        number of variables).
    
    See also
    --------
    swap_load_ascii_single_grid,swap_load_ascii_nultiple_grid,load_ascii_single_grid
    load_ascii_multiple_grid,swap_load_npy_grid,load_npy_grid,load_ascii_grid
    '''
    if path[-4:] == '.npy':
        return load_npy_grid(path,swap,swap_directory)
    else:
        if type(blocks) == tuple:
            if len(blocks) == 2:
                blocks = (blocks[0],blocks[1],1)
            elif len(blocks) == 1:
                blocks = (blocks[0],1,1)
            elif blocks == 3:
                pass
            else:
                print 'ERROR ON "load_grid": length of blocks tuple not recognized. Only length 1,2 or 3 accepted.'
                return False
            appex = determine_special_char_separator(path,at_least)
            header = appex[0]
            columns = appex[1]
            sep = appex[2]
            if type(dtype)==bool: dtype = check_dtype_on_special_char_file(path,header,columns,sep)
            return load_ascii_special_char_grid(path,blocks,header,dtype,columns,sep,swap,swap_directory)
        else:
            print 'ERROR ON "load_grid": blocks must be tuple of length 1,2 or 3.'
            return False
            
            
def load_ascii_special_char_point(path,coordinate_columns = (1,2,3),header=0,dtype='float32',sep = ' ',swap=False,swap_directory='TMP'):
    '''
    load_ascii_special_char_point(...)
        load_ascii_special_char_point(path,coordinate_columns,header,dtype,sep,swap,swap_directory)
        
        Loads a point ASCII (text) file. The separator for data in file
        is an argument.
        
    Parameters
    ----------
    path : string
        String with file path.
        
    coordinate_columns : tuple
        Tuple of integers giving the columns (starting in 1) of the X,Y and Z
        information. Default is (1,2,3) for (X,Y,Z).
        
    header : int
        Integer with the size of header rows. Default is zero.
        
    dtype : string
        String with data type. Default is float32.
        
    sep : string
        String with character that acts like separator. Default is ;
        
    swap : bool
        Boolean value indicating if output variable should be memory swap.
        Default is False.
        
    swap_directory : string
        String with directory to build the swap file (if swap is True).      
        
    Returns
    -------
    out: numpy array (or swap)
        2D numpy array with information on file (with X,Y,Z columns in 0,1,2).
    
    See also
    --------
    load_ascii_point, load_npy_point, load_point
    '''
    fid = open(path,'r')
    for i in xrange(header): fid.readline()
    point = np.loadtxt(fid,dtype=dtype,delimiter=sep)
    coordinate_columns = [coordinate_columns[0],coordinate_columns[1],coordinate_columns[2]]
    number = 3-coordinate_columns.count(0)
    appex_point = np.zeros((point.shape[0],point.shape[1]+number),dtype=dtype)
    counter = 0
    for i in xrange(len(coordinate_columns)):
        if coordinate_columns[i] != 0 and coordinate_columns[i]-1 < point.shape[1]:
            appex_point[:,counter] = point[:,coordinate_columns[i]-1]
        counter = counter + 1
    counter = 3
    for i in xrange(point.shape[0]):
        if i+1 not in coordinate_columns:
            appex_point[:,counter] = point[:,i]
            counter = counter + 1           
    fid.close()
    if swap:
        directory_list_of_files = os.listdir(swap_directory)
        chosen_name = 'point_'+str(np.random.randint(1000,9999))+'.dat'
        while chosen_name in directory_list_of_files:
            chosen_name = str(np.random.randint(1000,9999))+'.dat'
        mem_point = np.memmap(swap_directory+'\\'+chosen_name, dtype=point.dtype, mode='w+', shape=point.shape, order='F')
        mem_point[:] = appex_point[:]
        return mem_point
    else:
        return appex_point
            
def load_special_char_point(path,coordinate_columns,dtype='float32',sep=' ',swap=False,swap_directory='TMP',at_least=3):
    '''
    load_special_char_point(...)
        load_special_char_point(path,coordinate_columns,dtype,sep,swap,swap_directory,at_least)
        
        Loads a point ASCII (text) file. If file path ends in .npy its assumed
        its a numpy binary (NPY file). Tests to number and columns (whitespace
        separator) are made. The separator for data in file is an argument.
        
    Parameters
    ----------
    path : string
        String with file path.
        
    coordinate_columns : tuple
        Tuple of integers giving the columns (starting in 1) of the X,Y and Z
        information. Default is (1,2,3) for (X,Y,Z).
        
    dtype : string
        String with data type. Default is float32.
        
    sep : string
        String with character that acts like separator. Default is ;
        
    swap : bool
        Boolean value indicating if output variable should be memory swap.
        Default is False.
        
    swap_directory : string
        String with directory to build the swap file (if swap is True).    
        
    at_least : int
        Integer giving the number of lines in row which are digit. If it gets
        to this number than its assumed the information is no longer header
        information. Default is 3.
        
    Returns
    -------
    out: numpy array (or swap)
        2D numpy array with information on file (with X,Y,Z columns in 0,1,2).
    
    See also
    --------
    load_ascii_point, load_npy_point, load_point
    '''
    if path[-4:] == '.npy':
        return load_npy_point(path,swap,swap_directory)
    else:
        appex = determine_special_char_separator(path,at_least)
        header = appex[0]
        columns = appex[1]
        sep = appex[2]
        if type(dtype)==bool: dtype = check_dtype_on_special_char_file(path,header,columns,sep)
        return load_ascii_special_char_point(path,coordinate_columns,header,dtype,sep,swap,swap_directory)
        
def load_ascii_special_char_data(path,header=0,dtype='float32',sep = ' ',swap=False,swap_directory='TMP'):
    '''
    load_ascii_special_char_data(...)
        load_ascii_special_char_data(path,header,dtype,sep,swap,swap_directory)
        
        Loads a data (non-spatial) ASCII (text) file. The separator for data
        in file is an argument.
        
    Parameters
    ----------
    path : string
        String with file path.
        
    header : int
        Integer with the size of header rows.    
    
    dtype : string
        String with data type. Default is float32.
        
    sep : string
        String with character that acts like separator. Default is ;
        
    swap : bool
        Boolean value indicating if output variable should be memory swap.
        Default is False.
        
    swap_directory : string
        String with directory to build the swap file (if swap is True).    
        
    Returns
    -------
    out: numpy array
        2D numpy array with information on file.
    
    See also
    --------
    load_ascii_data, load_npy_point, load_data
    '''
    fid = open(path,'r')
    for i in xrange(header): fid.readline()
    data = np.loadtxt(fid,dtype=dtype,delimiter=sep)
    fid.close()
    if swap:
        directory_list_of_files = os.listdir(swap_directory)
        chosen_name = 'data_'+str(np.random.randint(1000,9999))+'.dat'
        while chosen_name in directory_list_of_files:
            chosen_name = str(np.random.randint(1000,9999))+'.dat'
        mem_data = np.memmap(swap_directory+'\\'+chosen_name, dtype=data.dtype, mode='w+', shape=data.shape, order='F')
        mem_data[:] = data[:]
        #del(point)
        return mem_data
    else:
        return data
        
def load_special_char_data(path,dtype='float32',swap=False,swap_directory='TMP',at_least=3):
    '''
    load_special_char_data(...)
        load_special_char_data(path,dtype,swap,swap_directory,at_least)
        
        Loads a data (non-spatial) file. If file ends with .npy its assumed
        its a numpy binary (NPY) file and load_npy_point is used. The separator
        for data in file is accepted if i whitespace or ; or , or _ .
        
    Parameters
    ----------
    path : string
        String with file path.  
    
    dtype : string
        String with data type. Default is float32.
        
    swap : bool
        Boolean value indicating if output variable should be memory swap.
        Default is False.
        
    swap_directory : string
        String with directory to build the swap file (if swap is True).
        
    at_least : int
        Integer giving the number of lines in row which are digit. If it gets
        to this number than its assumed the information is no longer header
        information. Default is 3.
        
    Returns
    -------
    out: numpy array
        2D numpy array with information on file (assumed no spatial information).
    
    See also
    --------
    load_ascii_data, load_npy_point, load_data
    '''
    if path[-4:] == '.npy':
        load_npy_point(path,swap,swap_directory)
    else:
        appex = determine_special_char_separator(path,at_least)
        header = appex[0]
        columns = appex[1]
        sep = appex[2]
        if type(dtype)==bool: dtype = check_dtype_on_special_char_file(path,header,columns,sep)
        return load_ascii_special_char_data(path,header,dtype,sep,swap,swap_directory)
       