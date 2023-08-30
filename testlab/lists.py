import sys
from random import randint

import matplotlib.pyplot as plt


def run():
    generators()


def generators():
    cubic_generator = (x**3 for x in range(10000))
    cubic_list = list(cubic_generator)
    print(f"Size of 'cubic_generator': {sys.getsizeof(cubic_generator):,} bytes.")
    print(f"Size of 'cubic_list': {sys.getsizeof(cubic_list):,} bytes.")

    small_generator = (x**2 for x in range(1000) if x ** 2 < 8000)
    small_list = generator_get_list(small_generator)
    print(f"List from hand-made list comprehension function: {small_list}")


def generator_get_list(generator):
    list = []
    try:
        while True:
            list.append(next(generator))
    except StopIteration:
        return list


def list_size_graph():
    my_list = []
    sizes = []

    for x in range(10000):
        my_list.append(randint(1, 100))

        sizes.append(sys.getsizeof(my_list))

        # sizes.append(
        #     (
        #         len(my_list),
        #         sys.getsizeof(my_list),
        #     )
        # )

    plt.style.use("seaborn")
    fig, ax = plt.subplots()

    x_values = list(range(1, 10001))
    ax.plot(x_values, sizes)

    ax.set_title("Size of list vs number of items in list", fontsize=24)
    ax.set_xlabel("Number of items", fontsize=14)
    ax.set_ylabel("Size of list (bytes)", fontsize=14)

    plt.show()
