# Copyright (c) Lynxi Technologies Co., Ltd. All rights reserved.
# Copyright (c) China Nanhu Academy of Electronics and Information Technology. All rights reserved.
import os
import numpy as np
from scipy.io import loadmat
import pandas
from scipy.ndimage.interpolation import zoom
import SimpleITK as sitk
from scipy.ndimage.morphology import binary_dilation,generate_binary_structure
from skimage.morphology import convex_hull_image
from multiprocessing import Pool
from functools import partial
from scipy.ndimage import rotate

def resample(imgs, spacing, new_spacing,order=2):
    if len(imgs.shape)==3:
        new_shape = np.round(imgs.shape * spacing / new_spacing)
        true_spacing = spacing * imgs.shape / new_shape
        resize_factor = new_shape / imgs.shape
        imgs = zoom(imgs, resize_factor, mode = 'nearest',order=order)
        return imgs, true_spacing
    elif len(imgs.shape)==4:
        n = imgs.shape[-1]
        newimg = []
        for i in range(n):
            slice = imgs[:,:,:,i]
            newslice,true_spacing = resample(slice,spacing,new_spacing)
            newimg.append(newslice)
        newimg=np.transpose(np.array(newimg),[1,2,3,0])
        return newimg,true_spacing
    else:
        raise ValueError('wrong shape')
def worldToVoxelCoord(worldCoord, origin, spacing):
     
    stretchedVoxelCoord = np.absolute(worldCoord - origin)
    voxelCoord = stretchedVoxelCoord / spacing
    return voxelCoord

def load_itk_image(filename):
    with open(filename) as f:
        contents = f.readlines()
        line = [k for k in contents if k.startswith('TransformMatrix')][0]
        transformM = np.array(line.split(' = ')[1].split(' ')).astype('float')
        transformM = np.round(transformM)
        if np.any( transformM!=np.array([1,0,0, 0, 1, 0, 0, 0, 1])):
            isflip = True
        else:
            isflip = False

    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)
     
    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))
     
    return numpyImage, numpyOrigin, numpySpacing,isflip

def process_mask(mask):
    convex_mask = np.copy(mask)
    for i_layer in range(convex_mask.shape[0]):
        mask1  = np.ascontiguousarray(mask[i_layer])
        if np.sum(mask1)>0:
            mask2 = convex_hull_image(mask1)
            if np.sum(mask2)>1.5*np.sum(mask1):
                mask2 = mask1
        else:
            mask2 = mask1
        convex_mask[i_layer] = mask2
    struct = generate_binary_structure(3,1)  
    dilatedMask = binary_dilation(convex_mask,structure=struct,iterations=10) 
    return dilatedMask


def lumTrans(img):
    lungwin = np.array([-1200.,600.])
    newimg = (img-lungwin[0])/(lungwin[1]-lungwin[0])
    newimg[newimg<0]=0
    newimg[newimg>1]=1
    newimg = (newimg*255).astype('uint8')
    return newimg     


def savenpy_luna(id, annos, filelist, luna_segment, luna_data,savepath,dvs_save_path,labels_dict):
    islabel = True
    isClean = True
    resolution = np.array([1,1,1])
    name = filelist[id]
    if name not in labels_dict:
        nod_label=0
    else:
        nod_label = labels_dict[name]
    sliceim,origin,spacing,isflip = load_itk_image(os.path.join(luna_data,name+'.mhd'))

    Mask,origin,spacing,isflip = load_itk_image(os.path.join(luna_segment,name+'.mhd'))
    if isflip:
        Mask = Mask[:,::-1,::-1]
    newshape = np.round(np.array(Mask.shape)*spacing/resolution).astype('int')
    m1 = Mask==3
    m2 = Mask==4
    Mask = m1+m2
    
    xx,yy,zz= np.where(Mask)
    box = np.array([[np.min(xx),np.max(xx)],[np.min(yy),np.max(yy)],[np.min(zz),np.max(zz)]])
    box = box*np.expand_dims(spacing,1)/np.expand_dims(resolution,1)
    box = np.floor(box).astype('int')
    margin = 5
    extendbox = np.vstack([np.max([[0,0,0],box[:,0]-margin],0),np.min([newshape,box[:,1]+2*margin],axis=0).T]).T

    this_annos = np.copy(annos[annos[:,0]==(name)])        

    if isClean:
        convex_mask = m1
        dm1 = process_mask(m1)
        dm2 = process_mask(m2)
        dilatedMask = dm1+dm2
        Mask = m1+m2

        extramask = dilatedMask ^ Mask
        bone_thresh = 210
        pad_value = 170

        if isflip:
            sliceim = sliceim[:,::-1,::-1]
            print('flip!')
        sliceim = lumTrans(sliceim)
        sliceim = sliceim*dilatedMask+pad_value*(1-dilatedMask).astype('uint8')
        bones = (sliceim*extramask)>bone_thresh
        sliceim[bones] = pad_value
        
        sliceim1,_ = resample(sliceim,spacing,resolution,order=1)
        sliceim2 = sliceim1[extendbox[0,0]:extendbox[0,1],
                    extendbox[1,0]:extendbox[1,1],
                    extendbox[2,0]:extendbox[2,1]]
        sliceim = sliceim2[np.newaxis,...]

    if islabel:
        this_annos = np.copy(annos[annos[:,0]==(name)])
        label = []
        if len(this_annos)>0:
            
            for c in this_annos:
                pos = worldToVoxelCoord(c[1:4][::-1],origin=origin,spacing=spacing)
                if isflip:
                    pos[1:] = Mask.shape[1:3]-pos[1:]
                label.append(np.concatenate([pos,[c[4]/spacing[1]]]))
            
        label = np.array(label)
        if len(label)==0:
            label2 = np.array([[0,0,0,0]])
        else:
            label2 = np.copy(label).T
            label2[:3] = label2[:3]*np.expand_dims(spacing,1)/np.expand_dims(resolution,1)
            label2[3] = label2[3]*spacing[1]/resolution[1]
            label2[:3] = label2[:3]-np.expand_dims(extendbox[:,0],1)
            label2 = label2[:4].T
        show_nodules(sliceim,label2,name,dvs_save_path,nod_label)
    print(name)

def preprocess_luna():
    luna_segment='data/Luna16/seg-lungs-LUNA16'
    savepath = 'data/Luna16/save/'
    luna_data = 'data/Luna16/'
    luna_label = 'data/Luna16/CSVFILES/annotations.csv'
    finished_flag = '.flag_preprocessluna'
    dvs_save_path='data/Luna16/dvs_save/'
    label_path='data/Luna16/CSVFILES/labels.csv'
    print('starting preprocessing luna')
    if not os.path.exists(finished_flag):
        annos = np.array(pandas.read_csv(luna_label))
        labels = pandas.read_csv(label_path)
        df = pandas.DataFrame(labels) 
        labels_dict = df.set_index('seriesuid')['class'].to_dict()  
        pool = Pool(processes=4)
        if not os.path.exists(savepath):
            os.mkdir(savepath)
        for setidx in range(10):
            print('process subset', setidx)
            filelist = [f.split('.mhd')[0] for f in os.listdir(luna_data+'subset'+str(setidx)) if f.endswith('.mhd') ]
            np.random.shuffle(filelist)
            split_idx = int(0.8 * len(filelist))
            train_files = filelist[:split_idx]
            test_files = filelist[split_idx:]

            train_savepath = os.path.join(dvs_save_path, 'train')
            test_savepath = os.path.join(dvs_save_path, 'test')
        
            if not os.path.exists(savepath+'subset'+str(setidx)):
                os.mkdir(savepath+'subset'+str(setidx))
            if not os.path.exists(dvs_save_path):
                os.mkdir(dvs_save_path)
            if not os.path.exists(train_savepath):
                os.makedirs(train_savepath)
            if not os.path.exists(test_savepath):
                os.makedirs(test_savepath)

            partial_savenpy_luna_train = partial(savenpy_luna, annos=annos, filelist=train_files,
                                                luna_segment=luna_segment, luna_data=luna_data + 'subset' + str(setidx) + '/',
                                                savepath=savepath+'subset'+str(setidx)+'/', dvs_save_path=train_savepath, labels_dict=labels_dict)
            
            partial_savenpy_luna_test = partial(savenpy_luna, annos=annos, filelist=test_files,
                                               luna_segment=luna_segment, luna_data=luna_data + 'subset' + str(setidx) + '/',
                                               savepath=savepath+'subset'+str(setidx)+'/', dvs_save_path=test_savepath, labels_dict=labels_dict)
            
            N_train = len(train_files)
            _ = pool.map(partial_savenpy_luna_train, range(N_train))
            
            N_test = len(test_files)
            _ = pool.map(partial_savenpy_luna_test, range(N_test))

        pool.close()
        pool.join()
    print('end preprocessing luna')

def resize_image(image, shape, mode='constant', constant_values=0):  
    from scipy.ndimage import zoom  
    factors = [n / float(o) for n, o in zip(shape, image.shape)]  
    return zoom(image, factors, mode=mode, cval=constant_values)

def show_nodules(ctdat,ctlab,name,dvs_save_path,nod_label):
    for idx in range(ctlab.shape[0]):
        if abs(ctlab[idx,0])+abs(ctlab[idx,1])+abs(ctlab[idx,2])+abs(ctlab[idx,3])==0: continue
        z, x, y = int(ctlab[idx,0]), int(ctlab[idx,1]), int(ctlab[idx,2])
  
        target_shape = (8, 32, 32)  
        data = ctdat[0]

        half_target_shape = np.array(target_shape) // 2  
        start_indices = np.maximum(0, np.array([z,x, y]) - half_target_shape)  
        end_indices = np.minimum(data.shape, np.array([z,x, y]) + half_target_shape)  
         
        padded_shape = np.array([end_indices[i] - start_indices[i] for i in range(3)])  
        if any(padded_shape < target_shape):  
            pad_before=abs(np.minimum(0,np.array([z,x, y]) - half_target_shape))
            pad_after =abs(np.minimum(0,data.shape-(np.array([z,x, y]) + half_target_shape)))
            sub_image = np.pad(  
                data[start_indices[0]:end_indices[0], start_indices[1]:end_indices[1], start_indices[2]:end_indices[2]],  
                ((pad_before[0], pad_after[0]), (pad_before[1], pad_after[1]), (pad_before[2], pad_after[2])),  
                mode='constant',  
                constant_values=170  
            ) 
        else:
            sub_image = data[start_indices[0]:end_indices[0], start_indices[1]:end_indices[1], start_indices[2]:end_indices[2]] 
        if 'test' in dvs_save_path:
              np.save(os.path.join(dvs_save_path, f"{nod_label}_{name}_{idx}.npy"), sub_image)
        else:
            augmentations = [  
                sub_image,  # original  
                rotate(sub_image, 90, axes=(1, 2), reshape=False),  # rotate 90 degrees  
                rotate(sub_image, 180, axes=(1, 2), reshape=False),  # rotate 180 degrees  
                np.flip(sub_image, axis=1)  # flip along x-axis  
            ]
            for i, augmented_image in enumerate(augmentations):
                if augmented_image.shape != target_shape:  
                    augmented_image = np.array(  
                        [resize_image(augmented_image[..., j], target_shape[1:], mode='constant', constant_values=170)  
                        for j in range(augmented_image.shape[-1])]  
                    )  
                    augmented_image = np.transpose(augmented_image, (1, 2, 3, 0)) 

                np.save(os.path.join(dvs_save_path, f"{nod_label}_{name}_{idx}_aug{i}.npy"), augmented_image)
if __name__=='__main__':
    preprocess_luna()
