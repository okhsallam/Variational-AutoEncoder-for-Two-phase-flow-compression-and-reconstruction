import torch
import torch.nn.functional as F

def vort_x_calc(U):

   # print("U.requires_grad:", U.requires_grad)  # Added by Omar 

    # Calculate gradients along the specified axes for each batch
    v_z = torch.gradient(U[:, 1, :, :], axis=2)[0]
    w_y = torch.gradient(U[:, 2, :, :], axis=1)[0]
    
    # Compute vorticity in the x direction
    vort_x = w_y - v_z
    
    return vort_x

def div_calc(U):
   # print("U.requires_grad:", U.requires_grad)  # Added by Omar 

    # Calculate gradients along the specified axes for each batch
    v_y = torch.gradient(U[:, 1, :, :], axis=1)[0]
    w_z = torch.gradient(U[:, 2, :, :], axis=2)[0]
    
    # Compute vorticity in the x direction
    div = v_y + w_z
    
    return div

def curvature_calc(alpha):
    # Calculate gradients in y and z directions
    grad_alpha_y = torch.gradient(alpha, dim=1)[0]
    grad_alpha_z = torch.gradient(alpha, dim=2)[0]
    
    # Calculate the magnitude of the gradient
    grad_alpha_mag = torch.sqrt(grad_alpha_y ** 2 + grad_alpha_z ** 2 + 1e-10)  # Add epsilon to avoid division by zero
    
    # Normalize the gradient to get the unit normal vector
    n_y = grad_alpha_y / grad_alpha_mag
    n_z = grad_alpha_z / grad_alpha_mag
    
    # Compute the divergence of the normalized gradient
    div_n_y = torch.gradient(n_y, dim=1)[0]
    div_n_z = torch.gradient(n_z, dim=2)[0]
    curvature = div_n_y + div_n_z
    
    return -curvature  # Apply the negative sign to get the curvature
