import polars as pl
import numpy as np
import time
import random

repetitions = 10000000
num_keys = 30
start_time = time.time()
dictionary_list = []
# for _ in range(repetitions):
#     dictionary_data = {str(k): random.random() for k in range(num_keys)}
#     dictionary_list.append(dictionary_data)

# end_time = time.time()
# print(
#     "Execution time for generation [list of dict (row store)] = %.6f seconds"
#     % (end_time - start_time)
# )
# start_time = time.time()
# df_final1 = pl.DataFrame(dictionary_list)
# end_time = time.time()
# print(
#     "Execution time for conversion to pandas [list of dict (row store)] = %.6f seconds"
#     % (end_time - start_time)
# )

start_time = time.time()
list_dictionnary = {str(k): [] for k in range(num_keys)}
for _ in range(repetitions):
    for k in range(num_keys):
        list_dictionnary[str(k)].append(random.random())

end_time = time.time()
print(
    "Execution time for generation [dict of list (column store) = %.6f seconds"
    % (end_time - start_time)
)
start_time = time.time()
df_final2 = pl.from_dict(list_dictionnary)
end_time = time.time()
print(
    "Execution time for conversion to pandas [dict of list (column store)] = %.6f seconds"
    % (end_time - start_time)
)
