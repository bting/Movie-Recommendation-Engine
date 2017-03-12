import numpy
from heapq import *

# return user_iuf_matrix, which row represent users, col represent movies
# The rating value is applied iuf
def get_iuf_matrix(train_matrix):
	iuf_matrix = [[0]*1000 for i in range(200)]
	movie_iufs = [0]*1000
	for i in range(1000):
		movie_freq = len([0 for user in train_matrix if user[i] != 0])
		if movie_freq != 0:
			movie_iuf = numpy.log2(1.0*200/movie_freq)
			movie_iufs[i] = movie_iuf
			for iuf_user in iuf_matrix:
				iuf_user[i] = user[i]*movie_iuf
	return movie_iufs, iuf_matrix

# return the average rating of users
def avg_ratingOfUser(train_matrix):
	users_avg = [0]*200
	for i in range(200):
		users_avg[i] = numpy.average([rating for rating in train_matrix[i] if rating > 0])
	return users_avg


# deal with some special rating
def normal_rating(rating):
	rating = int(round(rating))
	if rating < 1:
		rating = 1
	if rating > 5:
		rating = 5
	return rating

# find common element of two vector
def get_common(vector1, vector2):
	com_vector1 = []
	com_vector2 = []
	for i in range(len(vector1)):
		if vector1[i] != 0 and vector2[i] != 0:
			com_vector1.append(vector1[i])
			com_vector2.append(vector2[i])
	return com_vector1, com_vector2


# cosine_similarity = (V1*V2)/(len(V1)*len(V2)), return float in [-1, 1]
# deal differently when v1 and v2 only have one common element
def cosine_similar (vector1, vector2):
	v1, v2 = get_common (vector1, vector2)
	if len(v1) == 0:
		return 0
	elif len(v1) == 1:
		if abs(v1[0] - v2[0]) >= 3:
			return 0
		elif abs(v1[0] - v2[0]) == 2:
			return 0.5
		elif abs(v1[0] - v2[0]) == 1:
			return 0.8
		else:
			return 1

	product = numpy.dot(v1, v2)
	v1_len = numpy.linalg.norm(v1)
	v2_len = numpy.linalg.norm(v2)
	return product/(v1_len*v2_len)

# pearson_correlation = ((v1-AVG(v1) * (v2-AVG(v2))) / (len(v1-AVG(v1)) * len(v2-AVG(v2)))
# deal differently when v1 and v2 only have one common element
# if one of v1_mean_len and v2_mean_len is 0, calculate their cosine_similar 
def pearson_correlation(vector1, vector2, mean1, mean2):
	v1, v2 = get_common(vector1, vector2)
	if len(v1) == 0:
		return 0
	if len(v1) == 1:
		if abs(v1[0] - v2[0]) >= 3:
			return -1
		elif abs(v1[0] - v2[0]) == 2:
			return 0.5
		elif abs(v1[0] - v2[0]) == 1:
			return 0.8
		else:
			return 1
	v1_mean = numpy.array(v1) - mean1
	v2_mean = numpy.array(v2) - mean2

	product = numpy.dot(v1_mean, v2_mean)
	v1_mean_len = numpy.linalg.norm(v1_mean)
	v2_mean_len = numpy.linalg.norm(v2_mean)
	if v1_mean_len != 0 and v2_mean_len != 0:
		return round(product/(v1_mean_len*v2_mean_len))
	else:
		return cosine_similar(v1, v2)

def adjusted_movie_rating(train_matrix, users_avg):
	new_train_matrix = [[0]*1000 for i in range(200)]
	for i in range(len(train_matrix)):
		for j in range(len(train_matrix[i])):
			new_train_matrix[i][j] = train_matrix[i][j] - users_avg[i]
	return new_train_matrix

def adjusted_cosine_similar(train_matrix, user_avg, movie_id1, movie_id2):
	new_train_matrix = adjusted_movie_rating(train_matrix, users_avg)
	inversed_train_matrix = numpy.array(test_matrix).transpose()
	v1, v2 = get_common (inversed_train_matrix[mid1-1], inversed_train_matrix[mid2-1])
	if len(v1) == 0:
		return 0
	elif len(v1) == 1:
		if abs(v1[0] - v2[0]) >= 3:
			return 0
		elif abs(v1[0] - v2[0]) == 2:
			return 0.5
		elif abs(v1[0] - v2[0]) == 1:
			return 0.8
		else:
			return 1

	product = numpy.dot(v1, v2)
	v1_len = numpy.linalg.norm(v1)
	v2_len = numpy.linalg.norm(v2)
	return product/(v1_len*v2_len)
# get the topK similar weight
def get_topK_similar(sim_array, k):
	heap = []
	for x in sim_array:
		if len(heap) < k or x > heap[0]:
			heappush(heap, x)
			if len(heap) == k+1:
				heappop(heap)
	return heap

# implement case_modification new_Wa,u = Wa,u * (abs(Wa,u)**(p-1))
def case_modification(weight, p):
	new_weight = weight*(abs(weight)**(p-1))
	return new_weight

# implement dirichlet_smoothing Smoothing_Ru = (n/(b+n))*Ru + (b/(b+n))*Gu
def dirichlet_smoothing(rating, cnt_rating, global_rating, b):
	new_rating = (1.0*cnt_rating/(b+cnt_rating))*rating + (1.0*b/(b+cnt_rating))*global_rating
	return new_rating


