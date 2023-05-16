import numpy as np

def handle_raw_patches(raw_patches: dict):
    """
    Xử lý dict các patch, trong có key là tên label, value là list chứa các patch thuộc cùng 1 key 
    """
    handled_patches = {}
    for key in list(raw_patches.keys()):
        handled_patches[key] = []
        for patch in raw_patches[key]:
            for i in range(patch.shape[0]):
                handled_patches[key].append(patch[i])
    return handled_patches

def combine_patches(patches, image_size, patch_size, stride):
    """
    Ghép các patch 3D lại thành ảnh 3D ban đầu và lấy giá trị trung bình cho các đoạn overlap.
    
    Args:
        patches (list): Danh sách các patch có kích thước (n_patches, patch_size, patch_size, patch_size).
        image_size (tuple): Kích thước của ảnh 3D ban đầu.
        patch_size (tuple): Kích thước của patch.
        stride (tuple): Khoảng cách giữa các patch.
    
    Returns:
        ndarray: Ảnh 3D được ghép lại từ các patch, với kích thước là image_size.
    """
    # Tính số lượng patch theo chiều W, H, D.
    W = (image_size[0] - patch_size[0]) // stride[0] + 1
    H = (image_size[1] - patch_size[1]) // stride[1] + 1
    D = (image_size[2] - patch_size[2]) // stride[2] + 1
    print(W, H, D)

    n_patches = len(patches)
    if n_patches < W * H * D:
        raise ValueError("Number of patches is less than required.")
    
    # Tạo một ma trận trống với kích thước là image_size.
    image = np.zeros(image_size, dtype=np.float32)
    
    # Tính giá trị trung bình cho các đoạn overlap.
    overlap_count = np.zeros(image_size, dtype=np.float32)

    for wi, w in enumerate(range(0, image_size[0] - patch_size[0] + 1, stride[0])):
        for hi, h in enumerate(range(0, image_size[1] - patch_size[1] + 1, stride[1])):
            for di, d in enumerate(range(0, image_size[2] - patch_size[2] + 1, stride[2])):  
                # Tính tọa độ của patch trong ma trận của ảnh 3D ban đầu.
                patch_coord = np.array([w, h, d])
                
                # Tính vị trí của patch trong mảng các patch.
                patch_idx = wi * H * D + hi * D + di
                if patch_idx >= n_patches:
                    raise ValueError("Invalid patch index.")
                
                # Lấy giá trị của patch.
                patch = np.array(patches[patch_idx][0].cpu())
                
                # Tính vị trí của patch trong ma trận của ảnh 3D ban đầu.
                start_coord = patch_coord #- center_coord
                end_coord = start_coord + patch_size
                
                # Cập nhật giá trị cho ảnh 3D và ma trận overlap_count.
                image[start_coord[0]:end_coord[0], start_coord[1]:end_coord[1], start_coord[2]:end_coord[2]] += patch
                overlap_count[start_coord[0]:end_coord[0], start_coord[1]:end_coord[1], start_coord[2]:end_coord[2]] += 1
    
    # Tránh chia cho 0 bằng cách gán các giá trị bằng 1.
    overlap_count[overlap_count == 0] = 1

    # Tính giá trị trung bình cho các đoạn overlap.
    image /= overlap_count

    return image