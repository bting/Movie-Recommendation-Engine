import numpy
from util import *

# set global variable
p = 2.5
beta = 2
threshold = 0.75

# deal with input data, and store them in users
def generate_train_matrix(train_file):
	train_matrix = []
	row = []
	cnt = 0
	with open(train_file, 'r') as f:
		line = f.read()
		for x in line.split():
			cnt += 1
			row.append(int(x))
			if cnt == 1000:
				cnt = 0
				train_matrix.append(row)
				row = []
	return train_matrix

def generate_test_data(test_file, offindex):
	testing = open(test_file, 'r').read().strip().split('\n')
	input_data = []
	with open(test_file, 'r') as f:
		for line in f:
			input_data.append([int(x) for x in line.split()])
	#print input_data[:10]
	test_matrix = [[0] * 1000 for i in range(100)]
	for line in input_data:
		test_matrix[line[0]-offindex][line[1]-1] = line[2]
	return input_data, test_matrix

def main():
	train_matrix = generate_train_matrix('train.txt')
	input_data5, test5_matrix = generate_test_data('test5.txt', 201)
	input_data10, test10_matrix  = generate_test_data('test10.txt', 301)
	input_data20, test20_matrix  = generate_test_data('test20.txt', 401)

	'''
	read_file_calc_avg('result5_cosine.txt', 'result5_iufPearson.txt', 'file_avg5.txt')
	read_file_calc_avg('result10_cosine.txt', 'result10_iufPearson.txt', 'file_avg10.txt')
	read_file_calc_avg('result20_cosine.txt', 'result20_iufPearson.txt', 'file_avg20.txt')
	'''


    # get global_rating of train_martrix
	users_avg = avg_ratingOfUser(train_matrix)
	global_rating = numpy.average(users_avg)
	movie_iufs, iuf_matrix = get_iuf_matrix(train_matrix)

	
	args = [
		test5_matrix,
		input_data5,
		'result5_cosine.txt',
		train_matrix,
		201,
		'cosine',
		global_rating,
		movie_iufs, 
		iuf_matrix,
		users_avg
	]
	args2 = args[:]
	args2[0] = test10_matrix
	args2[1] = input_data10
	args2[2] = 'result10_cosine.txt'
	args2[4] = 301
	args3 = args[:]
	args3[0] = test20_matrix
	args3[1] = input_data20
	args3[2] = 'result20_cosine.txt'
	args3[4] = 401

	'''
	args[2] = 'result5_item_cosine.txt'
	args2[2] = 'result10_item_cosine.txt'
	args3[2] = 'result20_item_cosine.txt'
	get_all_itembased_rating(args)
	get_all_itembased_rating(args2)
	get_all_itembased_rating(args3)
	'''

	
	# get the ratings result based on cosine_similar
	get_all_rating(args)
	get_all_rating(args2)
	get_all_rating(args3)
	

	'''
	# get the rating result based on pearson_correlation
	args[5] = args2[5] = args3[5] = 'pearson'
	args[2] = 'result5_pearson.txt'
	args2[2] = 'result10_pearson.txt'
	args3[2] = 'result20_pearson.txt'
	get_all_rating(args)
	get_all_rating(args2)
	get_all_rating(args3)
	

	
	# get the rating result based on iuf_pearson_correlation
	args[5] = args2[5] = args3[5] = 'iufPearson'
	args[2] = 'result5_iufPearson.txt'
	args2[2] = 'result10_iufPearson.txt'
	args3[2] = 'result20_iufPearson.txt'
	get_all_rating(args)
	get_all_rating(args2)
	get_all_rating(args3)
	


	
	# get the rating result based on get_weight_iufPearson
	args[5] = args2[5] = args3[5] = 'weight_iufPearson'
	args[2] = 'result5_weight_iufPearson.txt'
	args2[2] = 'result10_weight_iufPearson.txt'
	args3[2] = 'result20_weight_iufPearson.txt'
	get_all_rating(args)
	get_all_rating(args2)
	get_all_rating(args3)
	'''
	
# get rating based on cosine_similarity
# if total_weight == 0, use the smoothing result
def get_cosine_rating(target_user, movie_id, train_matrix, global_rating):
	total_rating = 0;
	total_weight = 0;

	# new target array without 0
	new_target = [x for x in target_user if x > 0]
	target_avg = numpy.average(new_target)
	# choose the len of the whole target_user or new_target???
	target_cnt = len(new_target)
	
	smoothing_rating = dirichlet_smoothing(target_avg, target_cnt, global_rating, beta)

	for i in range(200):
		user = train_matrix[i]
		weight = cosine_similar(target_user, user)
		if p > 1:
			weight = case_modification(weight, p)

		if user[movie_id-1] != 0 and weight > threshold:
			total_weight += weight
			total_rating += weight*user[movie_id-1]

	target_rating = 0.0
	if total_weight > 0:
		target_rating = 1.0*total_rating/total_weight
	else:
		target_rating = target_avg
		#target_rating = smoothing_rating

	return normal_rating(target_rating)

# p is the case modification parameter; If p == 1, it would be the based pearson correlation
def get_pearson_rating(target_user, movie_id, train_matrix):
	total_rating = 0
	total_weight = 0

	target_avg = numpy.average([x for x in target_user if x > 0])
	for i in range(200):
		user = train_matrix[i]
		user_avg = numpy.average([x for x in user if x > 0])
		weight = pearson_correlation(target_user, user, target_avg, user_avg)

		if p > 1:
			weight = case_modification(weight, p)

		if user[movie_id-1] != 0 and abs(weight) > threshold:
			user_avg = numpy.average([x for x in user if x > 0])
			total_weight += abs(weight)
			total_rating += weight*(user[movie_id-1] - user_avg)

	target_rating = 0
	if total_weight > 0:
		target_rating = target_avg + (1.0*total_rating/total_weight)
	else:
		target_rating = target_avg

	return normal_rating(target_rating)

# p is the case modification parameter, if p == 1, it would be the based iuf_pearson_correlation
def get_iufPearson_rating(target_user, movie_id, train_matrix, movie_iufs, iuf_matrix):
	total_weight = 0
	total_rating = 0
	# apply iuf to targer_user_matrix, target_iuf = []
	target_iuf = [rate for rate in numpy.array(target_user) * numpy.array(movie_iufs)]
	# calculate the average rating of target_user, which is not applied iuf
	avg_target = numpy.average([rating for rating in target_user if rating > 0])

	for i in range(200):
		user = train_matrix[i]
		# get average rating of user, which is not applied iuf
		avg_user = numpy.average([rate for rate in user if rate > 0])
		# get the rating of user[i], which applied iuf, user_iuf = []
		user_iuf = iuf_matrix[i]
		# the weight calculated based on the ratings which be applied iuf
		weight = pearson_correlation(user_iuf, target_iuf, avg_target, avg_user)

		# whether to use case modification or not
		if p > 1:
			weight = case_modification(weight, p)

		if user[movie_id-1] != 0 and abs(weight) > threshold:
			total_rating += weight * (user[movie_id-1] - avg_user)
			total_weight += abs(weight)

	target_rating = 0
	if total_weight == 0:
		target_rating = avg_target
	else:
		target_rating = avg_target + (1.0*total_rating/total_weight)

	return normal_rating(target_rating)

# apply IUF to weight instead of original rating
def get_weight_iufPearson(target_user, movie_id, train_matrix, movie_iufs):
	total_weight = 0
	total_rating = 0

	avg_target = numpy.average([rating for rating in target_user if rating > 0])
	for i in range(200):
		user = train_matrix[i]
		avg_user = numpy.average([r for r in user if r > 0])
		weight = pearson_correlation(target_user, user, avg_target, avg_user)

		if p > 1:
			weight = case_modification(weight, p)
		if user[movie_id-1] != 0 and abs(weight) > threshold:
			total_rating += movie_iufs[movie_id-1] * weight * (user[movie_id-1] - avg_user)
			total_weight += abs(weight * movie_iufs[movie_id-1])

	target_rating = 0
	if total_weight != 0:
		target_rating = avg_target + (1.0*total_rating/total_weight)
	else:
		target_rating = avg_target
	return normal_rating(target_rating)

def get_adjusted_cosine_rating(train_matrix, users_avg, movie_id, user_index, inversed_train_matrix):
	total_weight = 0
	total_rating = 0	
	for i in range(1000):
		movie = inversed_train_matrix[i]
		if train_matrix[user_index][i] != 0:
			weight = cosine_similar(inversed_train_matrix[movie_id-1], movie)
			if weight > threshold:
				total_rating += train_matrix[user_index][i]*weight
				total_weight += weight

	target_rating = 0.0
	if total_weight > 0:
		target_rating = 1.0*total_rating/total_weight
	else:
		target_rating = 3

	return normal_rating(target_rating)


def get_all_rating(args):
	test_matrix, input_data, filename, train_matrix, offindex, tag, global_rating, movie_iufs, iuf_matrix, users_avg = args[:]
	output_file = open(filename, 'w')
	for user in input_data:
		if user[2] == 0:
			if tag == "cosine":
				user[2] = get_cosine_rating(test_matrix[user[0] - offindex], user[1], train_matrix, global_rating)
			elif tag == "pearson":
				user[2] = get_pearson_rating(test_matrix[user[0] - offindex], user[1], train_matrix)
			elif tag == "iufPearson":
				user[2] = get_iufPearson_rating(test_matrix[user[0] - offindex], user[1], train_matrix, movie_iufs, iuf_matrix)
			elif tag == "weight_iufPearson":
				user[2] = get_weight_iufPearson(test_matrix[user[0] - offindex], user[1], train_matrix, movie_iufs)

			output_file.write(' '.join(str(data)for data in user))
			output_file.write('\n')

	output_file.close()

def get_all_itembased_rating(args):
	test_matrix, input_data, filename, train_matrix, offindex, tag, global_rating, movie_iufs, iuf_matrix, users_avg = args[:]
	new_train_matrix = adjusted_movie_rating(train_matrix, users_avg)
	inversed_train_matrix = numpy.array(new_train_matrix).transpose()
	output_file = open(filename, 'w')
	for user in input_data:
		if user[2] == 0:
			user[2] = get_adjusted_cosine_rating(train_matrix, users_avg, user[1], user[0]-offindex, inversed_train_matrix)
			#if tag == "cosine":
				#user[2] = get_cosine_rating(test_matrix[user[1] - 1], user[0]-offindex, train_matrix, global_rating)
			#elif tag == "pearson":
				#user[2] = get_pearson_rating(test_matrix[user[1] - 1], user[0]-offindex, train_matrix)
			#elif tag == "iufPearson":
				#user[2] = get_iufPearson_rating(test_matrix[user[1] - 1], user[0], train_matrix, movie_iufs, iuf_matrix)
			#elif tag == "weight_iufPearson":
				#user[2] = get_weight_iufPearson(test_matrix[user[1] - 1], user[0]-offindex, train_matrix, movie_iufs)

			output_file.write(' '.join(str(data)for data in user))
			output_file.write('\n')

	output_file.close()

def read_result(filename):
	res = []
	with open(filename) as f:
		for line in f:
			data = [int(x) for x in line.split(' ')]
			res.append(data)
	return res

def read_file_calc_avg(file1, file2, outfile):
	res1 = read_result(file1)
	res2 = read_result(file2)
	with open(outfile, 'w') as f:
		for i in range(len(res1)):
			f.write(' '.join([str(res1[i][0]), str(res1[i][1]), str(normal_rating((res1[i][2]+res2[i][2])/2.0))]) + '\n')


main()




