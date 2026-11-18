import numpy as np

# Input .npy file
npy_path = "/home/hpc/mfpb/mfpb102h/master/SoftDwell/outputs/COC/1/4Det_24Res_1/test/predictions_nn.npy"

# Output .csv file
csv_path = "/home/hpc/mfpb/mfpb102h/master/SoftDwell/outputs/COC/1/4Det_24Res_1/test/predictions_nn.csv"

# Load data
data = np.load(npy_path)

print("Shape:", data.shape)
print("Dtype:", data.dtype)

# If 1D, save as one column
if data.ndim == 1:
    np.savetxt(csv_path, data, delimiter=",", fmt="%.10g")

# If 2D, save normally
elif data.ndim == 2:
    np.savetxt(csv_path, data, delimiter=",", fmt="%.10g")

# If 3D or higher, flatten except first dimension
else:
    data_2d = data.reshape(data.shape[0], -1)
    np.savetxt(csv_path, data_2d, delimiter=",", fmt="%.10g")

print(f"Saved CSV file to: {csv_path}")