import numpy as np

def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:
	#Write your code here and return a python list after reshaping by using numpy's tolist() method
	a = np.array(a)
	try:
		reshaped_matrix = a.reshape(new_shape[0], new_shape[1])
	except (ValueError, Exception):
		return []
	return reshaped_matrix.tolist()