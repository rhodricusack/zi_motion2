import numpy as np
import nibabel as nib
from nipype.interfaces import fsl
from nipype.interfaces.fsl import ImageStats

import shutil
import os
from matplotlib import pyplot as plt
import pandas as pd

import numpy as np
import pandas as pd

def rotation_matrix_zyx(rx, ry, rz, degrees=False):
    """
    Batch rotation matrices R = Rz @ Ry @ Rx.
    rx, ry, rz: arrays of shape (N,)
    Returns: (N, 3, 3)
    """
    rx = np.asarray(rx, dtype=float)
    ry = np.asarray(ry, dtype=float)
    rz = np.asarray(rz, dtype=float)

    if degrees:
        rx, ry, rz = np.deg2rad(rx), np.deg2rad(ry), np.deg2rad(rz)

    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    Rx = np.stack([
        np.stack([np.ones_like(cx), 0*cx,         0*cx        ], axis=-1),
        np.stack([0*cx,             cx,           -sx         ], axis=-1),
        np.stack([0*cx,             sx,            cx         ], axis=-1),
    ], axis=-2)

    Ry = np.stack([
        np.stack([ cy,              0*cy,          sy         ], axis=-1),
        np.stack([ 0*cy,            np.ones_like(cy), 0*cy    ], axis=-1),
        np.stack([-sy,              0*cy,          cy         ], axis=-1),
    ], axis=-2)

    Rz = np.stack([
        np.stack([ cz,             -sz,            0*cz       ], axis=-1),
        np.stack([ sz,              cz,            0*cz       ], axis=-1),
        np.stack([ 0*cz,            0*cz,          np.ones_like(cz)], axis=-1),
    ], axis=-2)

    return Rz @ Ry @ Rx

def adjust_pivot_df_shared_pivots(df, p_from, p_to, *, degrees=False,
                                  out_prefix="t_to", keep_vector_col=True):
    """
    Re-express each row's rigid transform so it's about p_to instead of p_from.
    The rotation (rx,ry,rz) stays the same; only translation changes.

    df must have columns: rx, ry, rz, tx, ty, tz
    p_from, p_to: numpy arrays of shape (3,), shared for all rows
    """
    # Ensure vectors
    p_from = np.asarray(p_from, dtype=float).reshape(3)
    p_to   = np.asarray(p_to,   dtype=float).reshape(3)

    # Extract data
    rx = df["rx"].to_numpy(dtype=float)
    ry = df["ry"].to_numpy(dtype=float)
    rz = df["rz"].to_numpy(dtype=float)
    t_from = df[["tx", "ty", "tz"]].to_numpy(dtype=float)  # (N,3)

    # Rotation matrices per row
    R = rotation_matrix_zyx(rx, ry, rz, degrees=degrees)  # (N,3,3)

    # Compute correction = (I - R) @ (p_from - p_to), broadcasted over rows
    I = np.eye(3)[None, :, :]                               # (1,3,3)
    delta = (p_from - p_to)[None, :]                        # (1,3)
    correction = np.einsum("nij,nj->ni", (I - R), np.broadcast_to(delta, (R.shape[0], 3)))

    t_to = t_from + correction

    # Write results
    df[f"{out_prefix}_x"] = t_to[:, 0]
    df[f"{out_prefix}_y"] = t_to[:, 1]
    df[f"{out_prefix}_z"] = t_to[:, 2]
    if keep_vector_col:
        df[out_prefix] = list(map(list, t_to.tolist()))

    return df

# -------- Example --------
# df = pd.DataFrame({
#     "rx":[0.2, 0.0], "ry":[0.4, 0.1], "rz":[0.8, 0.3],
#     "tx":[3.0,-1.0], "ty":[-1.0,2.0], "tz":[2.0,0.5],
# })
# p_from = np.array([10.0, -2.0, 5.0])
# p_to   = np.array([0.0,  0.0,  0.0])
# adjust_pivot_df_shared_pivots(df, p_from, p_to)
# print(df[["tx","ty","tz","t_to_x","t_to_y","t_to_z"]])

def ellipsoid_volume(
    shape=(32, 32, 64),
    center=(16.0, 16.0, 32.0),
    sizes=(10.0, 8.0, 14.0),
    rotations=(0.0, 0.0, 0.0),
    dtype=bool
):
    """
    Create a 3D array containing a filled rotated ellipsoid.

    Parameters
    ----------
    shape : tuple[int, int, int]
        Volume shape (Nx, Ny, Nz).
    center : tuple[float, float, float]
        Ellipsoid center (cx, cy, cz) in voxel coordinates.
    sizes : tuple[float, float, float]
        Semi-axes lengths (sx, sy, sz) in voxels (radii, not diameters).
    rotations : tuple[float, float, float]
        Euler angles (rx, ry, rz) in radians, applied Z->Y->X.
    dtype : numpy dtype
        Output dtype; use bool for a mask or np.uint8 for 0/1 bytes, etc.

    Returns
    -------
    vol : np.ndarray
        3D array of `shape` with the ellipsoid voxels set to True/1.
    """
    nx, ny, nz = shape
    cx, cy, cz = center
    sx, sy, sz = sizes
    rx, ry, rz = rotations

    # Coordinate grid (indexing='ij' so axes match array indices)
    x = np.arange(nx-1,-1,-1)[:, None, None] # x reversed to follow FSL convention
    y = np.arange(ny)[None, :, None]
    z = np.arange(nz)[None, None, :]

    # Shift to ellipsoid-centered coordinates
    X = x - np.ones((nx,ny,nz))*cx
    Y = y - np.ones((nx,ny,nz))*cy
    Z = z - np.ones((nx,ny,nz))*cz

    # Stack into 3x(N) for rotation
    P = np.stack((X.ravel(), Y.ravel(), Z.ravel()), axis=0)  # shape: (3, nx, ny, nz)

    # Build rotation matrix and rotate into ellipsoid's local frame
    R = rotation_matrix_zyx(-rx, ry, rz) # rx reversed to follow FSL convention
    # Use R^T to transform world coords into the ellipsoid's local axes
    Plocal = R.T @ P  # still (3, nx, ny, nz)
    Xp, Yp, Zp = Plocal[0], Plocal[1], Plocal[2]

    # Ellipsoid implicit function (<= 1 is inside)
    # (x/sx)^2 + (y/sy)^2 + (z/sz)^2 <= 1
    # Guard against zero sizes
    sx = float(sx) if sx != 0 else 1e-12
    sy = float(sy) if sy != 0 else 1e-12
    sz = float(sz) if sz != 0 else 1e-12

    F = (Xp / sx) ** 2 + (Yp / sy) ** 2 + (Zp / sz) ** 2

    mask = np.double(np.floor(F*4)%2 == 0)
    mask = (mask*2-1)*(np.sign(Xp) * np.sign(Yp) * np.sign(Zp))
    mask = (50 * (1+(mask > 0)))*(F <= 1.0)

    return mask.astype(dtype).reshape((nx,ny,nz))


def _euler_zyx_from_R(R, degrees=False):
    """
    Extract (rx, ry, rz) from R where R = Rz(rz) @ Ry(ry) @ Rx(rx).
    Returns angles in radians unless degrees=True.
    R: (..., 3, 3)
    """
    R = np.asarray(R, dtype=float)
    assert R.shape[-2:] == (3, 3)

    # ry = asin(-R[2,0])
    ry = np.arcsin(np.clip(-R[..., 2, 0], -1.0, 1.0))
    cy = np.cos(ry)

    # Handle near gimbal lock: |cy| ~ 0
    near_lock = np.isclose(cy, 0.0, atol=1e-8)

    rx = np.empty_like(ry)
    rz = np.empty_like(ry)

    # General case
    rx[~near_lock] = np.arctan2(R[..., 2, 1][~near_lock], R[..., 2, 2][~near_lock])
    rz[~near_lock] = np.arctan2(R[..., 1, 0][~near_lock], R[..., 0, 0][~near_lock])

    # Gimbal lock fallback: cy ~ 0, set rx=0 and fold into rz
    # When cy≈0, R ≈ Rz(rz) @ Ry(±π/2) @ Rx(rx) → columns lose independence.
    # Use alternative formulas:
    rx[near_lock] = 0.0
    sign = np.sign(-R[..., 2, 0][near_lock])  # sign of sin(ry)
    # If ry ≈ +π/2 (sign≈+1) or -π/2 (sign≈-1):
    # rz can be recovered from R[0,1] and R[1,1]
    rz[near_lock] = np.arctan2(-sign * R[..., 0, 1][near_lock], sign * R[..., 1, 1][near_lock])

    if degrees:
        return np.rad2deg(rx), np.rad2deg(ry), np.rad2deg(rz)
    return rx, ry, rz

def mcflirt_params_from_affines(affines, center, degrees=False, return_array=True):
    """
    Convert 4x4 affine(s) to MCFLIRT-style parameters: rx, ry, rz, tx, ty, tz.

    Parameters
    ----------
    affines : array-like, shape (4,4) or (N,4,4)
        Rigid (or near-rigid) transforms in the same coord system as `center` (typically mm).
    center : array-like, shape (3,)
        Image centre about which rotations are defined (same coords as `affines`).
        Example for world mm: c = M @ [(nx-1)/2,(ny-1)/2,(nz-1)/2,1]
    degrees : bool
        If True, return angles in degrees (MCFLIRT uses radians in .par).
    return_array : bool
        If True, returns (N,6) ndarray; else returns a dict of arrays.

    Returns
    -------
    params : ndarray shape (6,) or (N,6)  OR  dict with keys 'rx','ry','rz','tx','ty','tz'
    """
    A = np.asarray(affines, dtype=float)
    if A.ndim == 2:
        A = A[None, ...]  # (1,4,4)
    assert A.shape[-2:] == (4, 4), "affines must be (4,4) or (N,4,4)"

    c = np.asarray(center, dtype=float).reshape(3)
    R = A[:, :3, :3]
    d = A[:, :3,  3]

    # Extract Euler angles (Z-Y-X intrinsic)
    rx, ry, rz = _euler_zyx_from_R(R, degrees=degrees)

    # Solve for translations consistent with rotation about `center`
    # A[:3,3] = d = (I - R) c + t  =>  t = d - (I - R) c
    I = np.eye(3)[None, :, :]
    corr = (I - R) @ c  # (N,3)
    t = d - corr

    if return_array:
        out = np.column_stack([rx, ry, rz, t[:, 0], t[:, 1], t[:, 2]])
        return out[0] if out.shape[0] == 1 else out
    else:
        return dict(rx=rx, ry=ry, rz=rz, tx=t[:,0], ty=t[:,1], tz=t[:,2])

# -------- Convenience: compute world-centre from NIfTI header --------
def nifti_world_center(img):
    """
    Given a nibabel image, return the world-space centre coordinate (mm) that MCFLIRT uses.
    """
    shape = np.array(img.shape[:3], dtype=float)
    ijk_center = np.array([(shape[0]-1)/2.0, (shape[1]-1)/2.0, (shape[2]-1)/2.0, 1.0])
    M = img.affine  # voxel->world (mm)
    return (M @ ijk_center)[:3]

def params_from_affines(name, centre=None):
    pth = os.path.join(name + 'mcf.nii.gz.mat')
    ind = 0
    affines=[]
    while True:
        fn = os.path.join(pth,'MAT_%04d'%ind)
        if not os.path.exists(fn):
            break
        else:
            affines.append(np.loadtxt(fn))
        ind+=1
    return mcflirt_params_from_affines(affines, center=centre, degrees=False)

def generate_and_test(nii_path, name, pars,  ref_vol=0, voxsize=3, shape = (64,64,36), sizes = (20.0, 25.0, 17.0), clipz=None):
    # Generate ellipse data with motion parameters described in pars
    fn = os.path.join(nii_path,name)
    nvol = len(pars)
    if clipz is None:
        clipz=[0,shape[2],]
    vol = np.zeros( shape[:2] + (clipz[1]-clipz[0], nvol,) )
    for volind,par in enumerate(pars):
        wholevol = ellipsoid_volume(shape, center=par[3:],sizes=sizes, rotations=par[:3], dtype=np.uint16)
        vol[:,:,:,volind] = wholevol[:,:,clipz[0]:clipz[1]]
    mat = [[-voxsize,0,0,32*voxsize],[0,voxsize,0,-32*voxsize],[0,0,voxsize,-18*voxsize],[0,0,0,1]]
    img = nib.Nifti1Image(vol, mat)
    nib.save(img, fn + '.nii.gz')
    
    # Run MCFLIRT
    mcflt = fsl.MCFLIRT()
    mcflt.inputs.in_file = fn + '.nii.gz'
    mcflt.inputs.out_file = fn + 'mcf.nii.gz'
    mcflt.inputs.ref_vol = ref_vol
    mcflt.inputs.save_plots = True
    mcflt.inputs.save_mats = True
    print(mcflt.cmdline)
    print('Running MCFLIRT')
    res = mcflt.run()  
    print('...done')

    # Read in the MCFLIRT matrices and convert to parameters
    mcflt_pars_from_affines = params_from_affines(fn, centre = (0,0,0))
    mcflt_pars_from_affines = pd.DataFrame(mcflt_pars_from_affines, columns=['rx','ry','rz','tx','ty','tz'])


    fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(20,8))
    tlim=20
    rlim=0.45
    ax[0][0].plot(np.array(pars)[:,:3]-pars[0][:3])
    ax[0][0].set_title(f'{name}_true_rots')
    ax[0][0].set_ylim(-rlim,rlim)
    ax[1][0].plot(voxsize*(np.array(pars)[:,3:]-pars[0][3:]))
    ax[1][0].set_title(f'{name}_true_trans')
    ax[1][0].set_ylim(-tlim,tlim)
    ax[0][1].plot(mcflt_pars_from_affines[['rx','ry','rz']])
    ax[0][1].set_title(f'{name}_mcflirt_rots from affines')
    ax[0][1].set_ylim(-rlim,rlim)
    plt.legend(['rx','ry','rz'])
    ax[1][1].plot(mcflt_pars_from_affines[['tx','ty','tz']])
    ax[1][1].set_title(f'{name}_mcflirt_trans from affines')
    ax[1][1].set_ylim(-tlim,tlim)
    plt.legend(['tx','ty','tz'])

    # Origin of rotations affects values of translations but not rotations
    # For MCFLIRT/FLIRT affines, origin is voxel (0,0,0) of (clipped) volume
    # Shift to centre of full volume for consistency
    p_from_affine = (mat @ np.array([0,0,clipz[0],1]))[:3] 
    p_to_affine = (mat @ (np.concatenate((np.array(shape)/2,[1])).T))[:3]
    mcflt_pars_from_affines_adusted= adjust_pivot_df_shared_pivots(mcflt_pars_from_affines, p_from_affine, p_to_affine)
    
    ax[0][2].plot(mcflt_pars_from_affines_adusted[['rx','ry','rz']])
    ax[0][2].set_title(f'{name}_mcflirt_rot from affines adjusted')
    ax[0][2].set_ylim(-rlim,rlim)
    plt.legend(['rx','ry','rz'])
    ax[1][2].plot(mcflt_pars_from_affines_adusted[['t_to_x','t_to_y','t_to_z']])
    ax[1][2].set_title(f'{name}_mcflirt_trans from affines adusted')
    ax[1][2].set_ylim(-tlim,tlim)
    plt.legend(['t_to_x','t_to_y','t_to_z'])

    # Now repeat but using the mcflirt .par file
    # For MCFLIRT .pars, origin is voxel com of reference volume
    # Shift to centre of full volume for consistency
    mcflt_pars = pd.read_csv(fn + 'mcf.nii.gz.par', sep='\s+', header=None, names = ['rx','ry','rz','tx','ty','tz'])

    mcflt_pars[['rx']] = -mcflt_pars[['rx']] # flip x rot to match FSL convention

    ax[0][3].plot(mcflt_pars[['rx','ry','rz']])
    ax[0][3].set_title(f'{name}_mcflirt_rots')
    ax[0][3].set_ylim(-rlim,rlim)
    plt.legend(['rx','ry','rz'])
    ax[1][3].plot(mcflt_pars[['tx','ty','tz']])
    ax[1][3].set_title(f'{name}_mcflirt_trans')
    ax[1][3].set_ylim(-tlim,tlim)
    plt.legend(['tx','ty','tz'])

    # Calc centre of mass of mean used as pivot by MCFLIRT
    results = ImageStats(in_file=fn+'.nii.gz', split_4d=True, op_string='-C').run()
    p_from_voxels=results.outputs.out_stat[ref_vol]
    p_from_voxels[2] += clipz[0] # adjust for clipped slices
    p_from = (mat @ np.concatenate((p_from_voxels,[1])).T)[:3]

    # Shift to centre of full volume for consistency
    p_to = (mat @ (np.concatenate((np.array(shape)/2,[1])).T))[:3]

    mcflt_pars_adusted= adjust_pivot_df_shared_pivots(mcflt_pars, p_from, p_to)

    ax[0][4].plot(mcflt_pars_adusted[['rx','ry','rz']])
    ax[0][4].set_title(f'{name}_mcflirt_rot adjusted')
    ax[0][4].set_ylim(-rlim,rlim)
    plt.legend(['rx','ry','rz'])
    ax[1][4].plot(mcflt_pars_adusted[['t_to_x','t_to_y','t_to_z']])
    ax[1][4].set_title(f'{name}_mcflirt_trans adusted')
    ax[1][4].set_ylim(-tlim,tlim)
    plt.legend(['t_to_x','t_to_y','t_to_z'])

    fig.savefig(f'mcflirt_{name}.png')

if __name__ == "__main__":
    # Define your parameters
    
         # (sx, sy, sz) radii
    
    # Delete output directory
    nii_path = 'nii_files'
    shutil.rmtree(nii_path, ignore_errors=True)
    os.mkdir(nii_path)

    nvol = 20 # how many volumes in time series
    cz = 18 # centre used for ellipsoid
    clipz = [0,16] # chunk of bottom slices
    clipz2 = [20,36] # chunk of upper slices
    
    # x translations only
    pars_xtrans = [(0,0,0,  tx,32,cz) for tx in np.linspace(30,34,nvol)]
    generate_and_test(nii_path, 'xtrans', pars_xtrans)
    generate_and_test(nii_path, 'xtrans-clipz', pars_xtrans, clipz=clipz)

    # x rotations only
    pars_xrot = [(rx,0,0,  32,32,cz) for rx in np.linspace(-0.15,0.15,nvol)]
    generate_and_test(nii_path, 'xrot', pars_xrot)
    generate_and_test(nii_path, 'xrot-clipz', pars_xrot, clipz=clipz)
    generate_and_test(nii_path, 'xrot-clipz2', pars_xrot, clipz=clipz2)