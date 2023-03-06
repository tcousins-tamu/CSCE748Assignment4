# Import required libraries
import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
# Read source, target and mask for a given id
def Read(id, path = ""):
    source = plt.imread(path + "source_" + id + ".jpg")
    info = np.iinfo(source.dtype) # get information about the image type (min max values)
    source = source.astype(np.float32) / info.max # normalize the image into range 0 and 1
    target = plt.imread(path + "target_" + id + ".jpg")
    info = np.iinfo(target.dtype) # get information about the image type (min max values)
    target = target.astype(np.float32) / info.max # normalize the image into range 0 and 1
    mask   = plt.imread(path + "mask_" + id + ".jpg")
    info = np.iinfo(mask.dtype) # get information about the image type (min max values)
    mask = mask.astype(np.float32) / info.max # normalize the image into range 0 and 1

    return source, mask, target

# Adjust parameters, source and mask for negative offsets or out of bounds of offsets
def AlignImages(mask, source, target, offset):
    sourceHeight, sourceWidth, _ = source.shape
    targetHeight, targetWidth, _ = target.shape
    xOffset, yOffset = offset
    
    if (xOffset < 0):
        mask    = mask[abs(xOffset):, :]
        source  = source[abs(xOffset):, :]
        sourceHeight -= abs(xOffset)
        xOffset = 0
    if (yOffset < 0):
        mask    = mask[:, abs(yOffset):]
        source  = source[:, abs(yOffset):]
        sourceWidth -= abs(yOffset)
        yOffset = 0
    # Source image outside target image after applying offset
    if (targetHeight < (sourceHeight + xOffset)):
        sourceHeight = targetHeight - xOffset
        mask    = mask[:sourceHeight, :]
        source  = source[:sourceHeight, :]
    if (targetWidth < (sourceWidth + yOffset)):
        sourceWidth = targetWidth - yOffset
        mask    = mask[:, :sourceWidth]
        source  = source[:, :sourceWidth]
    
    maskLocal = np.zeros_like(target)
    maskLocal[xOffset:xOffset + sourceHeight, yOffset:yOffset + sourceWidth] = mask
    sourceLocal = np.zeros_like(target)
    sourceLocal[xOffset:xOffset + sourceHeight, yOffset:yOffset + sourceWidth] = source

    return sourceLocal, maskLocal

# Pyramid Blend, I don't know why this is here but I am going to ignore it for now
def PyramidBlend(source, mask, target):
    
    return source * mask + target * (1 - mask)

# Matrix Formation
def matrixParameters(source, mask, target, alpha = 1):
    """This Function generates the necessary components 
    create the sparse matrix components used for the poisson distribution.
    Split out from the other function due to length and complexity.

    Args:
        source (ndarray): Image that will be placed on target. Can have N channels.
        mask (ndarray): Image that specifies which elements from source will be placed on target. WILL ONLY USE FIRST CHANNEL.
        target (ndarray): The image that the cutout from the source will be placed onto. Can have N channels.
        alpha (int, optional): Alpha parameter for mixed gradient blending. Defaults to 1.

    Returns:
        tuple: Tuple Containing the Row indices, Column Indices, A (N Channels), and B (N Channels)
    """
    #Step One: Defininig the constants
    #Data and Const are separated by channels
    Cols, Rows, Data, Const = [],[],[], []
    for _ in range(target.shape[-1]):
        Data.append([])
        Const.append([])
    cnt = 0 
    
    #Step Two, Creating a 1D array of the information, will be used to create the 
    #sparse matrix. We assume source, mask, and target are appropriately sized
    for row in range(target.shape[0]):
        for col in range(target.shape[1]):
            #The mask may be 3 Dimensionsal, however we only want white, which is 1,1,1
            if mask[row, col, 0]<1: # I do not like that I had to add the 0 here, future iterations will take mask as 1D
                #These indices that are not used will have 1 in the matrix
                Cols.extend([cnt])
                Rows.extend([cnt])
                for chan in range(target.shape[-1]):
                    Data[chan].extend([1])
                    Const[chan].extend([target[row, col, chan]])
            #Non-empty mask portions
            else:
                Cols.extend([cnt, cnt, cnt, cnt, cnt])
                Rows.extend([cnt-1, cnt+1, cnt-source.shape[1], cnt+source.shape[1], cnt])
                for chan in range(target.shape[-1]):
                    #A array will have this added to it, in accordance with assignment notes.
                    Data[chan].extend([1, 1, 1, 1, -4])
                    
                    #Adding this section for the gradient mixing, with an alpha of 1 it has none. This equation was taken from assignment notes.
                    srcWeight = (source[row-1, col, chan] + source[row+1, col, chan] + source[row, col-1, chan] 
                        + source[row, col+1, chan] - 4*source[row, col, chan])
                    
                    tgtWeight = (target[row-1, col, chan] + target[row+1, col, chan] + target[row, col-1, chan] 
                        + target[row, col+1, chan] - 4*target[row, col, chan])
                    
                    Const[chan].extend([alpha*srcWeight + (1-alpha)*tgtWeight])
            cnt+=1
                
    #With this information, we can now shape an array to match the system of equations.
    return np.asarray(Rows, int), np.asarray(Cols, int), np.asarray(Data), np.asarray(Const)
# Poisson Blend
def PoissonBlend(source, mask, target, alpha = 1):
    """This function will perform the mixed gradient poisson blending 
    operation. All inputs MUST BE EQUIVALENTLY SIZED!

    Args:
        source (ndarray): Image that will be placed on target. Can have N channels.
        mask (ndarray): Image that specifies which elements from source will be placed on target. WILL ONLY USE FIRST CHANNEL.
        target (ndarray): The image that the cutout from the source will be placed onto. Can have N channels.
        alpha (int, optional): Alpha parameter for mixed gradient blending. Defaults to 1.

    Returns:
        ndarray: Output Image
    """
    #Pre-processing, we could check if the mask is on the edge or we could just pad to begin with anyways
    src = np.pad(source, pad_width=((1,1), (1,1), (0,0)), mode='constant')
    msk = np.pad(mask, pad_width=((1,1), (1,1), (0,0)), mode='constant')
    tgt = np.pad(target, pad_width=((1,1), (1,1), (0,0)), mode='constant')
    
    #Step One: Generating the sparse matrices (I used CSC here, but the underlying principles are same)
    Rows, Cols, Data, B = matrixParameters(src, msk, tgt, alpha)
    length = tgt.shape[0]*tgt.shape[1]

    #Step Two: Solving Ax=B. 
    #because I split this up by channels, it can work with grayscale and multi-channel images. 
    #This is useless for our demos, however I wanted to make it adaptive.
    solutions = []
    for chan in range(source.shape[-1]):
        tempSol = coo_matrix((Data[chan], (Cols, Rows)), shape = (length, length)) #This is also a sparse matrix
        tempSol = tempSol.tocsc() #Converts it to CSC form of sparse matrix that is a little faster for this
        tempSol = spsolve(tempSol, B[chan]) #solving for each channel
        solutions.append(tempSol.reshape(src.shape[:-1]))
        solutions[chan]=solutions[chan][1:-1, 1:-1] #Removing the padding I added earlier
        
    #Post-Processing: Flips the channels around to get it in the right order and clips
    solutions = np.asarray(solutions)
    solutions = np.moveaxis(solutions, 0, -1)

    solutions[np.where(solutions<0)]=0
    solutions[np.where(solutions>1)]=1
    
    return solutions

if __name__ == '__main__':
    # Setting up the input output paths
    inputDir = '../Images/'
    outputDir = '../Results/'
    
    # False for source gradient, true for mixing gradients.
    #This has been changed slightly in my implementation, as speciftying a non 0
    #Alpha is equilvalent 
    alpha = .5

    # Source offsets in target
    offsets = [[210, 10], [10, 28], [140, 80], [-40, 90], [60, 100], [20, 20], [-28, 88]]

    # main area to specify files and display blended image
    for index in range(len(offsets)):
        # Read data and clean mask
        source, maskOriginal, target = Read(str(index+1).zfill(2), inputDir)

        # Cleaning up the mask
        mask = np.ones_like(maskOriginal)
        mask[maskOriginal < 0.5] = 0

        # Align the source and mask using the provided offest
        source, mask = AlignImages(mask, source, target, offsets[index])

        
        ### The main part of the code ###
    
        # Implement the PoissonBlend function
        poissonOutput = PoissonBlend(source, mask, target, alpha)

        
        # Writing the result
                
        if alpha==1:
            plt.imsave("{}poisson_{}.jpg".format(outputDir, str(index+1).zfill(2)), poissonOutput)
        else:
            plt.imsave("{}poisson_{}_Mixing.jpg".format(outputDir, str(index+1).zfill(2)), poissonOutput)
