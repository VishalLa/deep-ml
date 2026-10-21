import numpy as np

def transform_matrix(
	A: list[list[int|float]], 
	T: list[list[int|float]], 
	S: list[list[int|float]]
) -> list[list[int|float]]:
	A, T, S = np.array(A), np.array(T), np.array(S)

	# check if T is invertible
	# if determent of T is zero then it is not invertible
	if (np.isclose(np.linalg.det(T), 0)):
		return -1 
	# check for S
	if (np.isclose(np.linalg.det(S), 0)):
		return  -1
	
	T_I = np.linalg.inv(T)

	transformed_matrix = T_I @ A @ S

	return transformed_matrix