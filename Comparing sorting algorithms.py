# Comparing algorithms

# 1. Bubble Sort ############################################################################################  

def bubble_sort(arr):

    arr_len = len(arr)

    for i in range(arr_len-1):
        flag = 0
        for j in range(0, arr_len-i-1):
            if arr[j] > arr[j+1]:
                arr[j+1], arr[j] = arr[j], arr[j+1]
                flag = 1
                if flag == 0:
                    break
    return arr

# 2. Selection Sort ############################################################################################  

def selection_sort(arr):
    n = len(arr)
    for i in range(n):
        min_index = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_index]:
                min_index = j
        arr[i], arr[min_index] = arr[min_index], arr[i]
        
# 3. Insertion Sort ############################################################################################  

def insertion_sort(arr):
    n = len(arr)
    for i in range(1, n):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
        
# 4. Merge sort ############################################################################################  

def mergeSort(arr):
	if len(arr) > 1:

		# Finding the mid of the array
		mid = len(arr)//2

		# Dividing the array elements
		L = arr[:mid]

		# into 2 halves
		R = arr[mid:]

		# Sorting the first half
		mergeSort(L)

		# Sorting the second half
		mergeSort(R)

		i = j = k = 0

		# Copy data to temp arrays L[] and R[]
		while i < len(L) and j < len(R):
			if L[i] <= R[j]:
				arr[k] = L[i]
				i += 1
			else:
				arr[k] = R[j]
				j += 1
			k += 1

		# Checking if any element was left
		while i < len(L):
			arr[k] = L[i]
			i += 1
			k += 1

		while j < len(R):
			arr[k] = R[j]
			j += 1
			k += 1

# Code to print the list
	
# 5. Quick sort ############################################################################################  


# Function to find the partition position
def partition(arr, low, high):

	# Choose the rightmost element as pivot
	pivot = arr[high]

	# Pointer for greater element
	i = low - 1

	# Traverse through all elements
	# compare each element with pivot
	for j in range(low, high):
		if arr[j] <= pivot:

			# If element smaller than pivot is found
			# swap it with the greater element pointed by i
			i = i + 1

			# Swapping element at i with element at j
			(arr[i], arr[j]) = (arr[j], arr[i])

	# Swap the pivot element with
	# the greater element specified by i
	(arr[i + 1], arr[high]) = (arr[high], arr[i + 1])

	# Return the position from where partition is done
	return i + 1


# Function to perform quicksort
def quicksort(arr, low, high):
	if low < high:

		# Find pivot element such that
		# element smaller than pivot are on the left
		# element greater than pivot are on the right
		pi = partition(arr, low, high)

		# Recursive call on the left of pivot
		quicksort(arr, low, pi - 1)

		# Recursive call on the right of pivot
		quicksort(arr, pi + 1, high)


		
# 6. Heap Sort ############################################################################################  


def heapify(arr, N, i):
	largest = i # Initialize largest as root
	l = 2 * i + 1	 # left = 2*i + 1
	r = 2 * i + 2	 # right = 2*i + 2

	# See if left child of root exists and is
	# greater than root
	if l < N and arr[largest] < arr[l]:
		largest = l

	# See if right child of root exists and is
	# greater than root
	if r < N and arr[largest] < arr[r]:
		largest = r

	# Change root, if needed
	if largest != i:
		arr[i], arr[largest] = arr[largest], arr[i] # swap

		# Heapify the root.
		heapify(arr, N, largest)

# The main function to sort an array of given size


def heapSort(arr):
	N = len(arr)

	# Build a maxheap.
	for i in range(N//2 - 1, -1, -1):
		heapify(arr, N, i)

	# One by one extract elements
	for i in range(N-1, 0, -1):
		arr[i], arr[0] = arr[0], arr[i] # swap
		heapify(arr, i, 0)


############### Comparing

import random
# Generate a list of 10000 random integers
arr = [random.randint(1, 10000) for i in range(10000)]
 
# Sort the list using each algorithm and time it
import time
 
start_time = time.time()
bubble_sort(arr.copy())
bubble_sort_time = time.time() - start_time
 
start_time = time.time()
selection_sort(arr.copy())
selection_sort_time = time.time() - start_time
 
start_time = time.time()
insertion_sort(arr.copy())
insertion_sort_time = time.time() - start_time

start_time = time.time()
mergeSort(arr.copy())
merge_sort_time = time.time() - start_time

start_time = time.time()
heapSort(arr.copy())
heap_sort_time = time.time() - start_time

print("Bubble Sort time:", bubble_sort_time)
print("Selection Sort time:", selection_sort_time)
print("Insertion Sort time:", insertion_sort_time)
print("Merge Sort time:", merge_sort_time)
print("Heap Sort time:", heap_sort_time)
