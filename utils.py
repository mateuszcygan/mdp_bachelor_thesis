# Return Boolean, if a certain number is included in array
def is_number_in_array(arr, num):
    for element in arr:
        if element == num:
            return True
    return False